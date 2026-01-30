"""
example_sqlitevec_mcp.py
Example: Build sqlite-vec vector database and serve via MCP

This is an APPLICATION EXAMPLE that demonstrates how to use the Parquet output
from the core pipeline (01-03) with a specific vector DB (sqlite-vec) and
serving method (MCP server).

Other examples could include:
- example_chroma.py: Using ChromaDB
- example_pinecone.py: Using Pinecone
- example_qdrant.py: Using Qdrant

Usage:
    # Build vector DB from chunked parquet
    uv run python example_sqlitevec_mcp.py build

    # Run MCP server (stdio mode for Claude Desktop, etc.)
    uv run python example_sqlitevec_mcp.py serve

    # Run MCP server (SSE mode for HTTP clients)
    uv run python example_sqlitevec_mcp.py serve --sse --port 8080

    # Build and serve in one command
    uv run python example_sqlitevec_mcp.py all

    # List supported embedding models
    uv run python example_sqlitevec_mcp.py --list-models

Supported embedding models:
- BAAI/bge-m3 (default, 1024 dim, multilingual)
- intfloat/multilingual-e5-large (1024 dim)
- sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast)
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
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import sqlite_vec
from sentence_transformers import SentenceTransformer

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent


# =============================================================================
# Directory Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "chunked_data"
OUTPUT_DIR = BASE_DIR / "vector_db"
DEFAULT_DB_PATH = OUTPUT_DIR / "vectors.db"

# Default embedding model
DEFAULT_MODEL = "BAAI/bge-m3"

# Model configurations (name -> dimension)
MODEL_CONFIGS = {
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-base": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


# =============================================================================
# Embedding Model
# =============================================================================

class LocalEmbeddingModel:
    """Local embedding model using sentence-transformers"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    def initialize(self) -> None:
        """Load the embedding model"""
        if self.model is not None:
            return

        print(f"🔄 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

        # Get embedding dimension
        test_embedding = self.model.encode(["test"], convert_to_numpy=True)
        self._dimension = test_embedding.shape[1]
        print(f"   ✅ Model loaded (dimension: {self._dimension})")

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            return MODEL_CONFIGS.get(self.model_name, 1024)
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        is_query: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.model is None:
            self.initialize()

        if self.model is None:
            raise RuntimeError("Model initialization failed")

        if not texts:
            return np.array([])

        # Handle model-specific prefixes
        processed_texts = texts
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            processed_texts = [prefix + t for t in texts]

        return self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# =============================================================================
# Vector DB Builder
# =============================================================================

class VectorDBBuilder:
    """sqlite-vec based vector DB builder"""

    def __init__(self, db_path: Path, model_name: str = DEFAULT_MODEL):
        self.db_path = db_path
        self.model_name = model_name
        self.conn: Optional[sqlite3.Connection] = None
        self.model: Optional[LocalEmbeddingModel] = None

    def _load_model(self) -> None:
        """Load embedding model"""
        self.model = LocalEmbeddingModel(self.model_name)
        self.model.initialize()

    def _check_model_compatibility(self) -> tuple[bool, Optional[str], Optional[int]]:
        """Check if existing DB was created with a different model."""
        if not self.db_path.exists():
            return (True, None, None)

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            cursor = conn.execute(
                "SELECT model_name, embedding_dim FROM model_info WHERE id = 1"
            )
            row = cursor.fetchone()

            if row is None:
                conn.close()
                return (True, None, None)

            existing_model, existing_dim = row
            expected_dim = MODEL_CONFIGS.get(self.model_name, 1024)

            # Check virtual table dimension
            try:
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_vectors'"
                )
                schema_row = cursor.fetchone()
                if schema_row and schema_row[0]:
                    match = re.search(r'FLOAT\[(\d+)\]', schema_row[0])
                    if match:
                        actual_vtable_dim = int(match.group(1))
                        if actual_vtable_dim != expected_dim:
                            conn.close()
                            return (False, existing_model, actual_vtable_dim)
            except Exception:
                pass

            conn.close()

            if existing_model == self.model_name and existing_dim == expected_dim:
                return (True, existing_model, existing_dim)

            if existing_dim != expected_dim:
                return (False, existing_model, existing_dim)

            return (False, existing_model, existing_dim)

        except Exception:
            return (True, None, None)

    def _prompt_rebuild(self, existing_model: str, existing_dim: int) -> bool:
        """Prompt user to confirm DB rebuild due to model change."""
        expected_dim = MODEL_CONFIGS.get(self.model_name, 1024)

        print("\n" + "=" * 50)
        print("⚠️  MODEL MISMATCH DETECTED")
        print("=" * 50)
        print(f"   Existing DB model: {existing_model} (dim={existing_dim})")
        print(f"   Requested model:   {self.model_name} (dim={expected_dim})")
        print("")
        print("   The vector database was created with a different embedding model.")
        print("   To use the new model, the existing database must be deleted.")
        print("")

        while True:
            response = input("   Delete and rebuild? [y/N]: ").strip().lower()
            if response in ("", "n", "no"):
                return False
            elif response in ("y", "yes"):
                return True
            else:
                print("   Please enter 'y' or 'n'.")

    def _delete_db(self) -> None:
        """Delete existing database file"""
        if self.db_path.exists():
            self.db_path.unlink()
            print(f"   🗑️  Deleted: {self.db_path}")

    def _init_db(self) -> None:
        """Initialize sqlite-vec database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        dim = self.model.dimension if self.model else 1024

        # Metadata table
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
                section_path TEXT,
                table_headers TEXT,
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

        # Model info table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_info (
                id INTEGER PRIMARY KEY,
                model_name TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TEXT
            )
        """)

        self.conn.execute("""
            INSERT OR REPLACE INTO model_info (id, model_name, embedding_dim, created_at)
            VALUES (1, ?, ?, ?)
        """, (self.model_name, dim, datetime.now().isoformat()))

        # Vector table
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{dim}]
            )
        """)

        # Indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks(source_file)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(domain)"
        )

        self.conn.commit()
        print(f"✅ DB initialized: {self.db_path} (dim={dim})")

    def _serialize_vector(self, vec: np.ndarray) -> bytes:
        """Convert numpy vector to sqlite-vec bytes"""
        return struct.pack(f"{len(vec)}f", *vec)

    def build(self, input_dir: Path = INPUT_DIR, force: bool = False) -> dict:
        """Build vector database"""
        input_dir = Path(input_dir)

        if not input_dir.exists():
            print(f"⚠️ Input directory not found: {input_dir}")
            return {"error": "Input directory not found"}

        parquet_files = list(input_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"⚠️ No parquet files found: {input_dir}")
            return {"error": "No parquet files found"}

        # Check model compatibility
        is_compatible, existing_model, existing_dim = self._check_model_compatibility()

        if not is_compatible:
            if existing_model and existing_dim:
                if force:
                    print(f"\n⚠️  Model mismatch: {existing_model} → {self.model_name}")
                    print("   --force specified, rebuilding database...")
                    self._delete_db()
                else:
                    if self._prompt_rebuild(existing_model, existing_dim):
                        self._delete_db()
                    else:
                        print("\n❌ Build cancelled. Use the same model or --force to rebuild.")
                        return {"error": "Model mismatch", "cancelled": True}

        print(f"\n📦 Building vector DB")
        print(f"   Model: {self.model_name}")
        print(f"   Input: {len(parquet_files)} parquet files")
        print("=" * 50)

        self._load_model()
        self._init_db()

        if self.conn is None:
            raise RuntimeError("Failed to connect to database")

        stats = {
            "total_chunks": 0,
            "embedded_chunks": 0,
            "skipped_chunks": 0,
            "files_processed": 0,
            "model": self.model_name,
        }

        now = datetime.now().isoformat()

        for i, pq_file in enumerate(parquet_files, 1):
            print(f"\n[{i}/{len(parquet_files)}] {pq_file.name}")

            try:
                df = pd.read_parquet(pq_file)
                print(f"   📖 Chunks: {len(df)}")

                # Check existing
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
                    print(f"   ⏭️ All chunks already exist")
                    stats["files_processed"] += 1
                    continue

                print(f"   🔍 New: {len(new_chunks)}, Skip: {len(existing)}")

                # Generate embeddings
                texts = new_chunks["chunk_text"].tolist()
                embeddings = self.model.encode(texts, batch_size=32, is_query=False) if self.model else np.array([])

                # Insert into DB
                for idx, (_, row) in enumerate(new_chunks.iterrows()):
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

                    vec_bytes = self._serialize_vector(embeddings[idx])
                    self.conn.execute(
                        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                        (row["chunk_id"], vec_bytes)
                    )

                    stats["embedded_chunks"] += 1

                self.conn.commit()
                stats["total_chunks"] += len(df)
                stats["files_processed"] += 1
                print(f"   ✅ Done")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 50)
        print(f"✅ Build complete")
        print(f"   📊 Total chunks: {stats['total_chunks']}")
        print(f"   🆕 Embedded: {stats['embedded_chunks']}")
        print(f"   ⏭️ Skipped: {stats['skipped_chunks']}")
        print(f"   💾 DB path: {self.db_path}")

        return stats

    def export_for_milvus(self, output_path: Path) -> Path:
        """Export to Milvus/Qdrant compatible parquet file"""
        if self.conn is None:
            raise RuntimeError("Database not initialized")

        print(f"\n📤 Exporting vector DB...")

        chunks_df = pd.read_sql_query("SELECT * FROM chunks ORDER BY id", self.conn)

        dim_result = self.conn.execute(
            "SELECT embedding_dim FROM model_info WHERE id = 1"
        ).fetchone()
        dim = dim_result[0] if dim_result else 1024

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
                vectors.append([0.0] * dim)

        chunks_df["embedding"] = vectors

        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_parquet(output_path, compression="zstd")

        print(f"✅ Export complete: {output_path}")
        print(f"   📊 Chunks: {len(chunks_df)}")
        print(f"   📐 Dimension: {dim}")

        return output_path

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


# =============================================================================
# Vector Search Service (for MCP Server)
# =============================================================================

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
            return

        start_time = time.time()
        self._log("Initialization starting...")

        if not self.db_path.exists():
            raise FileNotFoundError(f"Vector DB not found: {self.db_path}")

        self._log(f"Connecting to DB: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._log(f"DB connected ({time.time() - start_time:.2f}s)")

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

        processed_text = text
        if self.model_name and "e5" in self.model_name.lower():
            processed_text = "query: " + text

        return self.model.encode(
            [processed_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

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
        self.initialize()

        if self.conn is None:
            raise RuntimeError("DB not initialized")

        query_embedding = self._get_embedding(query)
        query_bytes = self._serialize_vector(query_embedding)

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

        total_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_docs = self.conn.execute(
            "SELECT COUNT(DISTINCT source_file) FROM chunks"
        ).fetchone()[0]

        domain_stats = self.conn.execute("""
            SELECT domain, COUNT(*) as count
            FROM chunks GROUP BY domain ORDER BY count DESC
        """).fetchall()

        type_stats = self.conn.execute("""
            SELECT chunk_type, COUNT(*) as count
            FROM chunks GROUP BY chunk_type ORDER BY count DESC
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
# MCP Server
# =============================================================================

def create_mcp_server(db_path: Path = DEFAULT_DB_PATH, preload: bool = True) -> Server:
    """Create MCP server"""

    server = Server("reconsidered-rag-search")
    search_service = VectorSearchService(db_path)

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
                        "query": {"type": "string", "description": "Search query (natural language)"},
                        "top_k": {"type": "integer", "description": "Number of results (default: 5)", "default": 5},
                        "domain": {"type": "string", "description": "Filter by domain (optional)"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_chunk",
                description="Look up the full content and metadata of a chunk by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string", "description": "Chunk ID"}},
                    "required": ["chunk_id"],
                },
            ),
            Tool(
                name="list_documents",
                description="List all indexed documents with their chunk counts.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_stats",
                description="Get vector database statistics including total chunks, "
                           "documents, and distribution by domain/type.",
                inputSchema={"type": "object", "properties": {}},
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
                    return [TextContent(type="text", text="No results found.")]

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
                    return [TextContent(type="text", text=f"Chunk not found: {chunk_id}")]

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
                    return [TextContent(type="text", text="No indexed documents.")]

                formatted = ["# Document List\n"]
                formatted.append("| File | Chunks | Domain | Language |")
                formatted.append("|------|--------|--------|----------|")
                for doc in docs:
                    formatted.append(
                        f"| {doc['source_file']} | {doc['chunk_count']} | "
                        f"{doc['domain']} | {doc['language']} |"
                    )

                return [TextContent(type="text", text="\n".join(formatted))]

            elif name == "get_stats":
                stats = search_service.get_stats()

                domains_str = "\n".join(f"  - {k}: {v}" for k, v in stats["domains"].items())
                types_str = "\n".join(f"  - {k}: {v}" for k, v in stats["chunk_types"].items())

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
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

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
        return JSONResponse({"status": "ok", "server": "reconsidered-rag-search"})

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


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Example: sqlite-vec vector DB + MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build     Build vector DB from chunked parquet files
  serve     Run MCP server (stdio mode by default)
  all       Build then serve

Examples:
  uv run python example_sqlitevec_mcp.py build
  uv run python example_sqlitevec_mcp.py serve
  uv run python example_sqlitevec_mcp.py serve --sse --port 8080
  uv run python example_sqlitevec_mcp.py all
  uv run python example_sqlitevec_mcp.py --list-models
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["build", "serve", "all"],
        default="serve",
        help="Command to run (default: serve)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory for build (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for build (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="vectors.db",
        help="Database filename (default: vectors.db)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Full path to vector DB (overrides --output-dir/--db-name)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild if model mismatch",
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="Export Milvus/Qdrant compatible parquet after build",
    )
    parser.add_argument(
        "--test-search",
        type=str,
        default=None,
        help="Run test search after build",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported embedding models",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run MCP server in SSE mode (default: stdio)",
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
        help="Don't preload model at server startup",
    )

    args = parser.parse_args()

    # List models
    if args.list_models:
        print("Supported embedding models:")
        print("-" * 50)
        for model, dim in MODEL_CONFIGS.items():
            default = " (default)" if model == DEFAULT_MODEL else ""
            print(f"  {model} (dim={dim}){default}")
        return 0

    # Determine DB path
    db_path = args.db_path if args.db_path else (args.output_dir / args.db_name)

    # Build command
    if args.command in ("build", "all"):
        builder = VectorDBBuilder(db_path, model_name=args.model)

        try:
            stats = builder.build(args.input_dir, force=args.force)

            if "error" in stats:
                return 1

            if args.export_parquet:
                export_path = args.output_dir / "vectors_export.parquet"
                builder.export_for_milvus(export_path)

            if args.test_search:
                print(f"\n🔍 Test search: '{args.test_search}'")
                print("-" * 50)

                # Need to reload with search service for proper query embedding
                search_service = VectorSearchService(db_path)
                results = search_service.search(args.test_search, top_k=3)
                for i, r in enumerate(results, 1):
                    print(f"\n[{i}] Similarity: {r['similarity']:.4f}")
                    print(f"    File: {r['source_file']}")
                    print(f"    Section: {' > '.join(r['section_path'])}")
                    print(f"    Content: {r['chunk_text'][:100]}...")
                search_service.close()

        finally:
            builder.close()

    # Serve command
    if args.command in ("serve", "all"):
        preload = not args.no_preload

        if args.sse:
            asyncio.run(run_sse_server(db_path, args.host, args.port, preload))
        else:
            asyncio.run(run_stdio_server(db_path, preload))

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        exit(130)
