"""
05_mcp_server.py
vector_db의 sqlite-vec 데이터베이스를 활용한 MCP 서버

지원 기능:
- 벡터 유사도 검색 (search)
- 청크 조회 (get_chunk)
- 문서 목록 (list_documents)
- 통계 조회 (get_stats)

실행 방법:
- stdio 모드: python 05_mcp_server.py
- SSE 모드: python 05_mcp_server.py --sse --port 8080
"""

import json
import sqlite3
import struct
import asyncio
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

import numpy as np
import sqlite_vec
import torch
from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    Tool,
    TextContent,
)


# 디렉터리 설정
BASE_DIR = Path(__file__).parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
CACHE_DIR = BASE_DIR / "cache" / "huggingface"
DEFAULT_DB_PATH = VECTOR_DB_DIR / "vectors.db"

# BGE-M3 설정
EMBEDDING_DIM = 1024


def get_device_info() -> tuple[str, bool]:
    """사용 가능한 디바이스 및 FP16 지원 여부 반환"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return device_name, True
    elif torch.backends.mps.is_available():
        return "Apple MPS", False
    else:
        return "CPU", False


class VectorSearchService:
    """벡터 검색 서비스"""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.model: Optional[Any] = None
        self.reranker: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> None:
        """서비스 초기화 (lazy loading)"""
        if self._initialized:
            return

        # DB 연결
        if not self.db_path.exists():
            raise FileNotFoundError(f"벡터 DB를 찾을 수 없습니다: {self.db_path}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # 모델 로드
        device_name, use_fp16 = get_device_info()
        print(f"[*] BGE-M3 모델 로딩 중... ({device_name})")
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, cache_dir=str(CACHE_DIR))
        print("[OK] 모델 로딩 완료")

        # 리랭커 모델 로드 (CPU 전용)
        print("[*] BGE 리랭커 모델 로딩 중... (CPU)")
        self.reranker = BGEM3FlagModel("BAAI/bge-reranker-large", use_fp16=False, cache_dir=str(CACHE_DIR))
        print("[OK] 리랭커 모델 로딩 완료")

        self._initialized = True

    def _get_embedding(self, text: str) -> np.ndarray:
        """텍스트의 Dense 임베딩 계산"""
        if self.model is None:
            raise RuntimeError("모델이 초기화되지 않았습니다")

        result = self.model.encode([text], batch_size=1)
        return result["dense_vecs"][0].astype(np.float32)

    def _serialize_vector(self, vec: np.ndarray) -> bytes:
        """numpy 벡터를 sqlite-vec용 bytes로 변환"""
        return struct.pack(f"{len(vec)}f", *vec)

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> list[dict]:
        """벡터 유사도 검색"""
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB가 초기화되지 않았습니다")

        # 쿼리 임베딩
        query_embedding = self._get_embedding(query)
        query_bytes = self._serialize_vector(query_embedding)

        # 벡터 검색
        limit = top_k * 2 if domain_filter else top_k
        results = self.conn.execute("""
            SELECT
                v.chunk_id,
                v.distance,
                c.chunk_text,
                c.source_file,
                c.heading_text,
                c.section_path,
                c.domain,
                c.chunk_type
            FROM chunk_vectors v
            JOIN chunks c ON v.chunk_id = c.chunk_id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
        """, (query_bytes, limit)).fetchall()

        # 결과 변환 및 필터링
        output = []
        for row in results:
            if domain_filter and row[6] != domain_filter:
                continue

            output.append({
                "chunk_id": row[0],
                "distance": float(row[1]),
                "similarity": 1 - float(row[1]),
                "chunk_text": row[2],
                "source_file": row[3],
                "heading_text": row[4],
                "section_path": json.loads(row[5]) if row[5] else [],
                "domain": row[6],
                "chunk_type": row[7],
            })

            if len(output) >= top_k:
                break

        # 리랭킹 적용 (BGE 리랭커 사용)
        if output and self.reranker is not None:
            candidate_texts = [item["chunk_text"] for item in output]
            scores = self.reranker.compute_score([[query, text] for text in candidate_texts])
            for i, item in enumerate(output):
                item["rerank_score"] = float(scores[i])
            # 리랭킹 점수로 재정렬 (높은 점수가 더 관련성 높음)
            output.sort(key=lambda x: x["rerank_score"], reverse=True)

        return output

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """특정 청크 조회"""
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB가 초기화되지 않았습니다")

        result = self.conn.execute("""
            SELECT
                chunk_id, content_hash, source_file, chunk_index,
                chunk_text, chunk_type, heading_level, heading_text,
                parent_heading, section_path, table_headers, table_row_count,
                domain, sub_domain, keywords, language, content_type
            FROM chunks
            WHERE chunk_id = ?
        """, (chunk_id,)).fetchone()

        if not result:
            return None

        return {
            "chunk_id": result[0],
            "content_hash": result[1],
            "source_file": result[2],
            "chunk_index": result[3],
            "chunk_text": result[4],
            "chunk_type": result[5],
            "heading_level": result[6],
            "heading_text": result[7],
            "parent_heading": result[8],
            "section_path": json.loads(result[9]) if result[9] else [],
            "table_headers": json.loads(result[10]) if result[10] else [],
            "table_row_count": result[11],
            "domain": result[12],
            "sub_domain": result[13],
            "keywords": result[14],
            "language": result[15],
            "content_type": result[16],
        }

    def list_documents(self) -> list[dict]:
        """문서 목록 조회"""
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB가 초기화되지 않았습니다")

        results = self.conn.execute("""
            SELECT
                source_file,
                COUNT(*) as chunk_count,
                domain,
                language
            FROM chunks
            GROUP BY source_file
            ORDER BY source_file
        """).fetchall()

        return [
            {
                "source_file": row[0],
                "chunk_count": row[1],
                "domain": row[2],
                "language": row[3],
            }
            for row in results
        ]

    def get_stats(self) -> dict:
        """DB 통계 조회"""
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB가 초기화되지 않았습니다")

        # 총 청크 수
        total_chunks = self.conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

        # 문서 수
        total_docs = self.conn.execute(
            "SELECT COUNT(DISTINCT source_file) FROM chunks"
        ).fetchone()[0]

        # 도메인별 통계
        domain_stats = self.conn.execute("""
            SELECT domain, COUNT(*) as count
            FROM chunks
            GROUP BY domain
            ORDER BY count DESC
        """).fetchall()

        # 청크 타입별 통계
        type_stats = self.conn.execute("""
            SELECT chunk_type, COUNT(*) as count
            FROM chunks
            GROUP BY chunk_type
            ORDER BY count DESC
        """).fetchall()

        return {
            "total_chunks": total_chunks,
            "total_documents": total_docs,
            "embedding_dimension": EMBEDDING_DIM,
            "db_path": str(self.db_path),
            "domains": {row[0]: row[1] for row in domain_stats},
            "chunk_types": {row[0]: row[1] for row in type_stats},
        }

    def close(self) -> None:
        """서비스 종료"""
        if self.conn:
            self.conn.close()
            self.conn = None
        self._initialized = False


# =============================================================================
# MCP 서버 정의
# =============================================================================

def create_mcp_server(db_path: Path = DEFAULT_DB_PATH) -> Server:
    """MCP 서버 생성"""

    server = Server("aipack-vector-search")
    search_service = VectorSearchService(db_path)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """사용 가능한 도구 목록"""
        return [
            Tool(
                name="search",
                description="벡터 유사도 기반으로 관련 문서 청크를 검색합니다. "
                           "자연어 쿼리를 입력하면 의미적으로 가장 유사한 청크들을 반환합니다.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색 쿼리 (자연어)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "반환할 결과 수 (기본값: 5)",
                            "default": 5,
                        },
                        "domain": {
                            "type": "string",
                            "description": "도메인 필터 (선택사항)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_chunk",
                description="특정 청크 ID로 청크의 전체 내용과 메타데이터를 조회합니다.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "청크 ID",
                        },
                    },
                    "required": ["chunk_id"],
                },
            ),
            Tool(
                name="list_documents",
                description="인덱싱된 모든 문서 목록과 각 문서의 청크 수를 조회합니다.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_stats",
                description="벡터 DB의 전체 통계를 조회합니다. "
                           "총 청크 수, 문서 수, 도메인별/타입별 분포 등을 반환합니다.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """도구 실행"""
        try:
            if name == "search":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 5)
                domain = arguments.get("domain")

                results = search_service.search(query, top_k, domain)

                if not results:
                    return [TextContent(
                        type="text",
                        text="검색 결과가 없습니다.",
                    )]

                # 결과 포맷팅
                formatted = []
                for i, r in enumerate(results, 1):
                    section_path = " > ".join(r["section_path"]) if r["section_path"] else "N/A"
                    formatted.append(
                        f"## [{i}] 유사도: {r['similarity']:.4f}\n"
                        f"- **파일**: {r['source_file']}\n"
                        f"- **섹션**: {section_path}\n"
                        f"- **타입**: {r['chunk_type']}\n"
                        f"- **ID**: {r['chunk_id']}\n\n"
                        f"```\n{r['chunk_text'][:500]}{'...' if len(r['chunk_text']) > 500 else ''}\n```\n"
                    )

                return [TextContent(
                    type="text",
                    text=f"# 검색 결과: '{query}'\n\n" + "\n".join(formatted),
                )]

            elif name == "get_chunk":
                chunk_id = arguments.get("chunk_id", "")
                chunk = search_service.get_chunk(chunk_id)

                if not chunk:
                    return [TextContent(
                        type="text",
                        text=f"청크를 찾을 수 없습니다: {chunk_id}",
                    )]

                section_path = " > ".join(chunk["section_path"]) if chunk["section_path"] else "N/A"

                return [TextContent(
                    type="text",
                    text=(
                        f"# 청크: {chunk_id}\n\n"
                        f"- **파일**: {chunk['source_file']}\n"
                        f"- **섹션**: {section_path}\n"
                        f"- **타입**: {chunk['chunk_type']}\n"
                        f"- **도메인**: {chunk['domain']}\n"
                        f"- **언어**: {chunk['language']}\n\n"
                        f"## 내용\n\n```\n{chunk['chunk_text']}\n```"
                    ),
                )]

            elif name == "list_documents":
                docs = search_service.list_documents()

                if not docs:
                    return [TextContent(
                        type="text",
                        text="인덱싱된 문서가 없습니다.",
                    )]

                formatted = ["# 문서 목록\n"]
                formatted.append("| 파일 | 청크 수 | 도메인 | 언어 |")
                formatted.append("|------|--------|--------|------|")
                for doc in docs:
                    formatted.append(
                        f"| {doc['source_file']} | {doc['chunk_count']} | "
                        f"{doc['domain']} | {doc['language']} |"
                    )

                return [TextContent(
                    type="text",
                    text="\n".join(formatted),
                )]

            elif name == "get_stats":
                stats = search_service.get_stats()

                domains_str = "\n".join(
                    f"  - {k}: {v}" for k, v in stats["domains"].items()
                )
                types_str = "\n".join(
                    f"  - {k}: {v}" for k, v in stats["chunk_types"].items()
                )

                return [TextContent(
                    type="text",
                    text=(
                        f"# 벡터 DB 통계\n\n"
                        f"- **총 청크 수**: {stats['total_chunks']}\n"
                        f"- **총 문서 수**: {stats['total_documents']}\n"
                        f"- **임베딩 차원**: {stats['embedding_dimension']}\n"
                        f"- **DB 경로**: {stats['db_path']}\n\n"
                        f"## 도메인별 분포\n{domains_str}\n\n"
                        f"## 청크 타입별 분포\n{types_str}"
                    ),
                )]

            else:
                return [TextContent(
                    type="text",
                    text=f"알 수 없는 도구: {name}",
                )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"오류 발생: {str(e)}",
            )]

    return server


# =============================================================================
# 서버 실행
# =============================================================================

async def run_stdio_server(db_path: Path) -> None:
    """stdio 모드로 서버 실행"""
    server = create_mcp_server(db_path)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_sse_server(db_path: Path, host: str, port: int) -> None:
    """SSE 모드로 서버 실행"""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    import uvicorn

    server = create_mcp_server(db_path)
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    async def handle_messages(request):
        await sse_transport.handle_post_message(
            request.scope, request.receive, request._send
        )

    async def health_check(request):
        return JSONResponse({"status": "ok", "server": "aipack-vector-search"})

    app = Starlette(
        routes=[
            Route("/health", health_check),
            Route("/sse", handle_sse),
            Route("/messages/", handle_messages, methods=["POST"]),
        ],
    )

    print(f"[START] MCP SSE Server: http://{host}:{port}")
    print(f"   - Health: http://{host}:{port}/health")
    print(f"   - SSE: http://{host}:{port}/sse")
    print(f"   - Messages: http://{host}:{port}/messages/")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="벡터 검색 MCP 서버를 실행합니다."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"벡터 DB 경로 (기본값: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="SSE 모드로 실행 (기본값: stdio 모드)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="SSE 서버 호스트 (기본값: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="SSE 서버 포트 (기본값: 8080)",
    )

    args = parser.parse_args()

    if args.sse:
        asyncio.run(run_sse_server(args.db_path, args.host, args.port))
    else:
        asyncio.run(run_stdio_server(args.db_path))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[서버 종료]")
        exit(0)
