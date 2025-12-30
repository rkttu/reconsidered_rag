"""
04_build_vector_db.py
Module that reads parquet files from chunked_data, generates BGE-M3 embeddings,
and compiles them into a local vector database based on sqlite-vec

Features:
- BGE-M3 Dense vector (1024-dimension) based search
- Utilizes sqlite-vec extension (vector similarity search)
- Stores metadata and vectors together
- Parquet export portable to Milvus/Qdrant, etc.
"""

import json
import sqlite3
import struct
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite_vec
import torch
from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]


# 디렉터리 설정
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "chunked_data"
OUTPUT_DIR = BASE_DIR / "vector_db"

# BGE-M3 설정
EMBEDDING_DIM = 1024  # BGE-M3 Dense 벡터 차원


def get_device_info() -> tuple[str, bool]:
    """Return available device and FP16 support status"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return device_name, True
    elif torch.backends.mps.is_available():
        return "Apple MPS", False
    else:
        return "CPU", False


class VectorDBBuilder:
    """sqlite-vec based vector DB builder"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.model: Optional[Any] = None

    def _load_model(self) -> None:
        """BGE-M3 모델 로드"""
        device_name, use_fp16 = get_device_info()

        if use_fp16:
            print(f"🔄 BGE-M3 모델 로딩 중... (GPU: {device_name}, FP16)")
        else:
            print(f"🔄 BGE-M3 모델 로딩 중... ({device_name}, FP32)")

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)
        print("✅ 모델 로딩 완료")

    def _init_db(self) -> None:
        """sqlite-vec DB 초기화"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # 메타데이터 테이블
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                content_hash TEXT,
                source_file TEXT,
                chunk_index INTEGER,
                chunk_text TEXT,
                chunk_type TEXT,
                heading_level INTEGER,
                heading_text TEXT,
                parent_heading TEXT,
                section_path TEXT,  -- JSON 배열
                table_headers TEXT,  -- JSON 배열
                table_row_count INTEGER,
                domain TEXT,
                sub_domain TEXT,
                keywords TEXT,
                language TEXT,
                content_type TEXT,
                version INTEGER,
                created_at TEXT,
                updated_at TEXT,
                embedded_at TEXT
            )
        """)

        # 벡터 테이블 (sqlite-vec 가상 테이블)
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{EMBEDDING_DIM}]
            )
        """)

        # 인덱스
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source_file
            ON chunks(source_file)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_domain
            ON chunks(domain)
        """)

        self.conn.commit()
        print(f"✅ DB 초기화 완료: {self.db_path}")

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트의 Dense 임베딩 계산"""
        if not texts or self.model is None:
            return np.array([])

        result = self.model.encode(texts, batch_size=32)
        return result["dense_vecs"].astype(np.float32)

    def _serialize_vector(self, vec: np.ndarray) -> bytes:
        """numpy 벡터를 sqlite-vec용 bytes로 변환"""
        return struct.pack(f"{len(vec)}f", *vec)

    def build(self, input_dir: Path = INPUT_DIR) -> dict:
        """
        벡터 DB 빌드

        Args:
            input_dir: chunked_data 디렉터리

        Returns:
            빌드 통계
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            print(f"⚠️ 입력 디렉터리가 없습니다: {input_dir}")
            return {"error": "Input directory not found"}

        parquet_files = list(input_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"⚠️ parquet 파일이 없습니다: {input_dir}")
            return {"error": "No parquet files found"}

        print(f"\n📦 벡터 DB 빌드 시작")
        print(f"   입력: {len(parquet_files)}개 parquet 파일")
        print("=" * 50)

        # 모델 및 DB 초기화
        self._load_model()
        self._init_db()

        # 타입 가드: conn이 None이면 오류
        if self.conn is None:
            raise RuntimeError("DB 연결에 실패했습니다")

        stats = {
            "total_chunks": 0,
            "embedded_chunks": 0,
            "skipped_chunks": 0,
            "files_processed": 0,
        }

        now = datetime.now().isoformat()

        for i, pq_file in enumerate(parquet_files, 1):
            print(f"\n[{i}/{len(parquet_files)}] {pq_file.name}")

            try:
                df = pd.read_parquet(pq_file)
                print(f"   📖 청크 수: {len(df)}")

                # 이미 존재하는 chunk_id 확인
                chunk_ids = df["chunk_id"].tolist()
                placeholders = ",".join(["?"] * len(chunk_ids))
                existing = set(
                    row[0] for row in self.conn.execute(
                        f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})",
                        chunk_ids
                    ).fetchall()
                )

                new_chunks = df[~df["chunk_id"].isin(existing)]
                stats["skipped_chunks"] += len(existing)

                if new_chunks.empty:
                    print(f"   ⏭️ 모든 청크가 이미 존재함")
                    stats["files_processed"] += 1
                    continue

                print(f"   🔍 새 청크: {len(new_chunks)}개, 스킵: {len(existing)}개")

                # 임베딩 생성
                texts = new_chunks["chunk_text"].tolist()
                embeddings = self._get_embeddings(texts)

                # DB에 삽입
                for idx, (_, row) in enumerate(new_chunks.iterrows()):
                    # 메타데이터 테이블
                    # section_path와 table_headers가 numpy 배열일 수 있으므로 list로 변환
                    section_path = row.get("section_path", [])
                    if hasattr(section_path, "tolist"):
                        section_path = section_path.tolist()
                    table_headers = row.get("table_headers", [])
                    if hasattr(table_headers, "tolist"):
                        table_headers = table_headers.tolist()

                    self.conn.execute("""
                        INSERT OR REPLACE INTO chunks (
                            chunk_id, content_hash, source_file, chunk_index,
                            chunk_text, chunk_type, heading_level, heading_text,
                            parent_heading, section_path, table_headers, table_row_count,
                            domain, sub_domain, keywords, language, content_type,
                            version, created_at, updated_at, embedded_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["chunk_id"],
                        row.get("content_hash", ""),
                        row.get("source_file", ""),
                        row.get("chunk_index", 0),
                        row["chunk_text"],
                        row.get("chunk_type", ""),
                        row.get("heading_level", 0),
                        row.get("heading_text", ""),
                        row.get("parent_heading", ""),
                        json.dumps(section_path, ensure_ascii=False),
                        json.dumps(table_headers, ensure_ascii=False),
                        row.get("table_row_count", 0),
                        row.get("domain", ""),
                        row.get("sub_domain", ""),
                        row.get("keywords", ""),
                        row.get("language", ""),
                        row.get("content_type", ""),
                        row.get("version", 1),
                        row.get("created_at", now),
                        row.get("updated_at", now),
                        now,
                    ))

                    # 벡터 테이블
                    vec_bytes = self._serialize_vector(embeddings[idx])
                    self.conn.execute(
                        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                        (row["chunk_id"], vec_bytes)
                    )

                    stats["embedded_chunks"] += 1

                self.conn.commit()
                stats["total_chunks"] += len(df)
                stats["files_processed"] += 1
                print(f"   ✅ 완료")

            except Exception as e:
                print(f"   ❌ 오류: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 50)
        print(f"✅ 빌드 완료")
        print(f"   📊 총 청크: {stats['total_chunks']}")
        print(f"   🆕 임베딩 생성: {stats['embedded_chunks']}")
        print(f"   ⏭️ 스킵: {stats['skipped_chunks']}")
        print(f"   💾 DB 위치: {self.db_path}")

        return stats

    def export_for_milvus(self, output_path: Path) -> Path:
        """
        Milvus/Qdrant 등으로 이식 가능한 Parquet 파일 내보내기

        벡터를 float32 배열로 포함하여 다른 DB에서 직접 import 가능
        """
        if self.conn is None:
            raise RuntimeError("DB가 초기화되지 않았습니다")

        print(f"\n📤 벡터 DB 내보내기 중...")

        # 모든 데이터 조회
        chunks_df = pd.read_sql_query(
            "SELECT * FROM chunks ORDER BY id", self.conn
        )

        # 벡터 조회 및 변환
        vectors = []
        for chunk_id in chunks_df["chunk_id"]:
            result = self.conn.execute(
                "SELECT embedding FROM chunk_vectors WHERE chunk_id = ?",
                (chunk_id,)
            ).fetchone()

            if result:
                vec_bytes = result[0]
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                vectors.append(vec.tolist())
            else:
                vectors.append([0.0] * EMBEDDING_DIM)

        chunks_df["embedding"] = vectors

        # Parquet 저장 (벡터 포함)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_parquet(output_path, compression="zstd")

        print(f"✅ 내보내기 완료: {output_path}")
        print(f"   📊 청크 수: {len(chunks_df)}")
        print(f"   📐 벡터 차원: {EMBEDDING_DIM}")

        return output_path

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        벡터 유사도 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            domain_filter: 도메인 필터 (선택)

        Returns:
            검색 결과 리스트
        """
        if self.conn is None or self.model is None:
            raise RuntimeError("DB 또는 모델이 초기화되지 않았습니다")

        # 쿼리 임베딩
        query_embedding = self._get_embeddings([query])[0]
        query_bytes = self._serialize_vector(query_embedding)

        # 벡터 검색 (sqlite-vec knn 쿼리)
        # 필터링을 위해 더 많이 가져옴
        limit = top_k * 2
        results = self.conn.execute("""
            SELECT
                v.chunk_id,
                v.distance,
                c.chunk_text,
                c.source_file,
                c.heading_text,
                c.section_path,
                c.domain
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
                "distance": row[1],
                "similarity": 1 - row[1],  # cosine distance → similarity
                "chunk_text": row[2],
                "source_file": row[3],
                "heading_text": row[4],
                "section_path": json.loads(row[5]) if row[5] else [],
                "domain": row[6],
            })

            if len(output) >= top_k:
                break

        return output

    def close(self) -> None:
        """DB 연결 종료"""
        if self.conn:
            self.conn.close()
            self.conn = None


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="chunked_data를 sqlite-vec 벡터 DB로 컴파일합니다."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"입력 디렉터리 (기본값: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"출력 디렉터리 (기본값: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="vectors.db",
        help="DB 파일명 (기본값: vectors.db)",
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="Milvus/Qdrant 이식용 Parquet 파일 내보내기",
    )
    parser.add_argument(
        "--test-search",
        type=str,
        default=None,
        help="빌드 후 테스트 검색 수행",
    )

    args = parser.parse_args()

    db_path = args.output_dir / args.db_name
    builder = VectorDBBuilder(db_path)

    try:
        # 빌드
        stats = builder.build(args.input_dir)

        if "error" in stats:
            return 1

        # Parquet 내보내기
        if args.export_parquet:
            export_path = args.output_dir / "vectors_export.parquet"
            builder.export_for_milvus(export_path)

        # 테스트 검색
        if args.test_search:
            print(f"\n🔍 테스트 검색: '{args.test_search}'")
            print("-" * 50)

            results = builder.search(args.test_search, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] 유사도: {r['similarity']:.4f}")
                print(f"    파일: {r['source_file']}")
                print(f"    섹션: {' > '.join(r['section_path'])}")
                print(f"    내용: {r['chunk_text'][:100]}...")

        return 0

    finally:
        builder.close()


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[중단됨]")
        exit(130)
