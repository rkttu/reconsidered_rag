"""
05_mcp_server.py
MCP Server with Mini RAG using sqlite-vec vector database

Features:
- Vector similarity search with local embedding models
- Chunk lookup and document listing
- Statistics and database info
- Mini RAG: answer questions using retrieved context

Supported modes:
- stdio: python 05_mcp_server.py
- SSE: python 05_mcp_server.py --sse --port 8080
"""

import sys
import io

# Force UTF-8 encoding on Windows (for emoji output)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace'
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace'
    )

import json
import sqlite3
import struct
import asyncio
from pathlib import Path
from typing import Any, Optional
import time

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    Tool,
    TextContent,
)


# Directory configuration
BASE_DIR = Path(__file__).parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
DEFAULT_DB_PATH = VECTOR_DB_DIR / "vectors.db"


class VectorSearchService:
    """Vector search service with local embedding model"""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.model: Optional[SentenceTransformer] = None
        self.model_name: Optional[str] = None
        self.embedding_dim: int = 1024
        self._initialized = False

    def _log(self, message: str) -> None:
        """Log with timestamp (to stderr)"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)

    def initialize(self) -> None:
        """Initialize service (lazy loading)"""
        if self._initialized:
            self._log("Already initialized, skipping")
            return

        start_time = time.time()
        self._log("Initialization starting...")

        # Connect to DB
        if not self.db_path.exists():
            raise FileNotFoundError(f"Vector DB not found: {self.db_path}")

        self._log(f"Connecting to DB: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._log(f"DB connected ({time.time() - start_time:.2f}s)")

        # Get model info from DB
        model_info = self.conn.execute(
            "SELECT model_name, embedding_dim FROM model_info WHERE id = 1"
        ).fetchone()

        if model_info:
            self.model_name = model_info[0]
            self.embedding_dim = model_info[1]
            self._log(f"Model info from DB: {self.model_name} (dim={self.embedding_dim})")
        else:
            self.model_name = "BAAI/bge-m3"
            self._log(f"No model info in DB, using default: {self.model_name}")

        # Load embedding model
        model_start = time.time()
        self._log(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        self._log(f"Model loaded ({time.time() - model_start:.2f}s)")

        self._initialized = True
        self._log(f"Initialization complete (total {time.time() - start_time:.2f}s)")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text (query mode)"""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self._log(f"Computing embedding: '{text[:50]}...'")
        start = time.time()

        # Handle query prefix for E5 models
        processed_text = text
        if self.model_name and "e5" in self.model_name.lower():
            processed_text = "query: " + text

        result = self.model.encode(
            [processed_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        self._log(f"Embedding computed ({time.time() - start:.2f}s)")
        return result

    def _serialize_vector(self, vec: np.ndarray) -> bytes:
        """Convert numpy vector to sqlite-vec bytes"""
        return struct.pack(f"{len(vec)}f", *vec)

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> list[dict]:
        """Vector similarity search"""
        self._log(f"search() called: query='{query[:50]}...', top_k={top_k}")
        search_start = time.time()

        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB not initialized")

        # Query embedding
        query_embedding = self._get_embedding(query)
        query_bytes = self._serialize_vector(query_embedding)

        # Vector search
        self._log("Searching vector DB...")
        db_start = time.time()
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
        self._log(f"DB search complete: {len(results)} results ({time.time() - db_start:.2f}s)")

        # Convert and filter results
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

        self._log(f"search() complete: {len(output)} results (total {time.time() - search_start:.2f}s)")
        return output

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Get specific chunk by ID"""
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB not initialized")

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
        """List all indexed documents"""
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB not initialized")

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
        """Get database statistics"""
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB not initialized")

        # Total chunks
        total_chunks = self.conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

        # Total documents
        total_docs = self.conn.execute(
            "SELECT COUNT(DISTINCT source_file) FROM chunks"
        ).fetchone()[0]

        # Domain stats
        domain_stats = self.conn.execute("""
            SELECT domain, COUNT(*) as count
            FROM chunks
            GROUP BY domain
            ORDER BY count DESC
        """).fetchall()

        # Chunk type stats
        type_stats = self.conn.execute("""
            SELECT chunk_type, COUNT(*) as count
            FROM chunks
            GROUP BY chunk_type
            ORDER BY count DESC
        """).fetchall()

        return {
            "total_chunks": total_chunks,
            "total_documents": total_docs,
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "db_path": str(self.db_path),
            "domains": {row[0]: row[1] for row in domain_stats},
            "chunk_types": {row[0]: row[1] for row in type_stats},
        }

    def close(self) -> None:
        """Close service"""
        if self.conn:
            self.conn.close()
            self.conn = None
        self._initialized = False


# =============================================================================
# MCP Server Definition
# =============================================================================

def create_mcp_server(db_path: Path = DEFAULT_DB_PATH, preload: bool = True) -> Server:
    """Create MCP server

    Args:
        db_path: Path to vector database
        preload: If True, preload model at server start
    """

    server = Server("reconsidered-rag-search")
    search_service = VectorSearchService(db_path)

    # Preload model at server start
    if preload:
        print("[Server] Preloading model... (one-time only)", file=sys.stderr, flush=True)
        search_service.initialize()
        print("[Server] Model loaded, ready for requests", file=sys.stderr, flush=True)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="search",
                description="Search for relevant document chunks using vector similarity. "
                           "Input a natural language query to find semantically similar content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (natural language)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                        "domain": {
                            "type": "string",
                            "description": "Filter by domain (optional)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_chunk",
                description="Look up the full content and metadata of a chunk by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "Chunk ID",
                        },
                    },
                    "required": ["chunk_id"],
                },
            ),
            Tool(
                name="list_documents",
                description="List all indexed documents with their chunk counts.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_stats",
                description="Get vector database statistics including total chunks, "
                           "documents, and distribution by domain/type.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Execute tool"""
        try:
            if name == "search":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 5)
                domain = arguments.get("domain")

                results = search_service.search(query, top_k, domain)

                if not results:
                    return [TextContent(
                        type="text",
                        text="No results found.",
                    )]

                # Format results
                formatted = []
                for i, r in enumerate(results, 1):
                    section_path = " > ".join(r["section_path"]) if r["section_path"] else "N/A"
                    formatted.append(
                        f"## [{i}] Similarity: {r['similarity']:.4f}\n"
                        f"- **File**: {r['source_file']}\n"
                        f"- **Section**: {section_path}\n"
                        f"- **Type**: {r['chunk_type']}\n"
                        f"- **ID**: {r['chunk_id']}\n\n"
                        f"```\n{r['chunk_text'][:500]}{'...' if len(r['chunk_text']) > 500 else ''}\n```\n"
                    )

                return [TextContent(
                    type="text",
                    text=f"# Search Results: '{query}'\n\n" + "\n".join(formatted),
                )]

            elif name == "get_chunk":
                chunk_id = arguments.get("chunk_id", "")
                chunk = search_service.get_chunk(chunk_id)

                if not chunk:
                    return [TextContent(
                        type="text",
                        text=f"Chunk not found: {chunk_id}",
                    )]

                section_path = " > ".join(chunk["section_path"]) if chunk["section_path"] else "N/A"

                return [TextContent(
                    type="text",
                    text=(
                        f"# Chunk: {chunk_id}\n\n"
                        f"- **File**: {chunk['source_file']}\n"
                        f"- **Section**: {section_path}\n"
                        f"- **Type**: {chunk['chunk_type']}\n"
                        f"- **Domain**: {chunk['domain']}\n"
                        f"- **Language**: {chunk['language']}\n\n"
                        f"## Content\n\n```\n{chunk['chunk_text']}\n```"
                    ),
                )]

            elif name == "list_documents":
                docs = search_service.list_documents()

                if not docs:
                    return [TextContent(
                        type="text",
                        text="No indexed documents.",
                    )]

                formatted = ["# Document List\n"]
                formatted.append("| File | Chunks | Domain | Language |")
                formatted.append("|------|--------|--------|----------|")
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
                        f"# Vector DB Statistics\n\n"
                        f"- **Total Chunks**: {stats['total_chunks']}\n"
                        f"- **Total Documents**: {stats['total_documents']}\n"
                        f"- **Embedding Model**: {stats['embedding_model']}\n"
                        f"- **Embedding Dimension**: {stats['embedding_dimension']}\n"
                        f"- **DB Path**: {stats['db_path']}\n\n"
                        f"## Distribution by Domain\n{domains_str}\n\n"
                        f"## Distribution by Chunk Type\n{types_str}"
                    ),
                )]

            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}",
            )]

    return server


# =============================================================================
# Server Execution
# =============================================================================

async def run_stdio_server(db_path: Path, preload: bool = True) -> None:
    """Run server in stdio mode"""
    server = create_mcp_server(db_path, preload=preload)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_sse_server(db_path: Path, host: str, port: int, preload: bool = True) -> None:
    """Run server in SSE mode"""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    import uvicorn

    server = create_mcp_server(db_path, preload=preload)
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
        return JSONResponse({
            "status": "ok",
            "server": "reconsidered-rag-search",
        })

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
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run vector search MCP server"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Vector DB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run in SSE mode (default: stdio mode)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="SSE server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="SSE server port (default: 8080)",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Don't preload model at startup",
    )

    args = parser.parse_args()

    preload = not args.no_preload

    if args.sse:
        asyncio.run(run_sse_server(args.db_path, args.host, args.port, preload))
    else:
        asyncio.run(run_stdio_server(args.db_path, preload))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server stopped]")
        exit(0)
