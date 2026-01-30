"""
03_build_vector_db.py
Build sqlite-vec vector database from chunked parquet files

Features:
- Local embedding models (BGE-M3, multilingual-e5-large, etc.)
- No external API required - fully offline capable
- sqlite-vec extension for vector similarity search
- Portable export to Milvus/Qdrant-compatible parquet

Supported embedding models:
- BAAI/bge-m3 (default, 1024 dim, multilingual)
- intfloat/multilingual-e5-large (1024 dim)
- sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast)
"""

import json
import sqlite3
import struct
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import sqlite_vec
from sentence_transformers import SentenceTransformer


# Directory configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "chunked_data"
OUTPUT_DIR = BASE_DIR / "vector_db"

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
            # Use predefined config or default
            return MODEL_CONFIGS.get(self.model_name, 1024)
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        is_query: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            is_query: If True, add query prefix for asymmetric models
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            self.initialize()

        # Type guard after initialization
        if self.model is None:
            raise RuntimeError("Model initialization failed")

        if not texts:
            return np.array([])

        # BGE-M3 and E5 models use instruction prefixes
        processed_texts = texts
        if "bge" in self.model_name.lower():
            if is_query:
                # For queries, no prefix needed in BGE-M3
                pass
            # For documents, no prefix needed in BGE-M3
        elif "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            processed_texts = [prefix + t for t in texts]

        return self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )


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

    def _init_db(self) -> None:
        """Initialize sqlite-vec database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Get embedding dimension
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

        # Insert or update model info
        self.conn.execute("""
            INSERT OR REPLACE INTO model_info (id, model_name, embedding_dim, created_at)
            VALUES (1, ?, ?, ?)
        """, (self.model_name, dim, datetime.now().isoformat()))

        # Vector table (sqlite-vec virtual table)
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{dim}]
            )
        """)

        # Indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source_file
            ON chunks(source_file)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_domain
            ON chunks(domain)
        """)

        self.conn.commit()
        print(f"✅ DB initialized: {self.db_path} (dim={dim})")

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for text list"""
        if not texts or self.model is None:
            return np.array([])

        return self.model.encode(texts, batch_size=32, is_query=False)

    def _serialize_vector(self, vec: np.ndarray) -> bytes:
        """Convert numpy vector to sqlite-vec bytes"""
        return struct.pack(f"{len(vec)}f", *vec)

    def build(self, input_dir: Path = INPUT_DIR) -> dict:
        """
        Build vector database

        Args:
            input_dir: Directory containing chunked parquet files

        Returns:
            Build statistics
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            print(f"⚠️ Input directory not found: {input_dir}")
            return {"error": "Input directory not found"}

        parquet_files = list(input_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"⚠️ No parquet files found: {input_dir}")
            return {"error": "No parquet files found"}

        print(f"\n📦 Building vector DB")
        print(f"   Model: {self.model_name}")
        print(f"   Input: {len(parquet_files)} parquet files")
        print("=" * 50)

        # Initialize model and DB
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

                # Check existing chunk_ids
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
                embeddings = self._get_embeddings(texts)

                # Insert into DB
                for idx, (_, row) in enumerate(new_chunks.iterrows()):
                    # Handle numpy arrays in section_path/table_headers
                    section_path = row.get("section_path", [])
                    if hasattr(section_path, "tolist"):
                        section_path = section_path.tolist()
                    table_headers = row.get("table_headers", [])
                    if hasattr(table_headers, "tolist"):
                        table_headers = table_headers.tolist()

                    # Metadata table
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

                    # Vector table
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
        """
        Export to Milvus/Qdrant compatible parquet file

        Includes vectors as float32 arrays for direct import
        """
        if self.conn is None:
            raise RuntimeError("Database not initialized")

        print(f"\n📤 Exporting vector DB...")

        # Query all data
        chunks_df = pd.read_sql_query(
            "SELECT * FROM chunks ORDER BY id", self.conn
        )

        # Get embedding dimension
        dim_result = self.conn.execute(
            "SELECT embedding_dim FROM model_info WHERE id = 1"
        ).fetchone()
        dim = dim_result[0] if dim_result else 1024

        # Query and convert vectors
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

        # Save as parquet (with vectors)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_parquet(output_path, compression="zstd")

        print(f"✅ Export complete: {output_path}")
        print(f"   📊 Chunks: {len(chunks_df)}")
        print(f"   📐 Dimension: {dim}")

        return output_path

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Vector similarity search

        Args:
            query: Search query
            top_k: Number of results to return
            domain_filter: Filter by domain (optional)

        Returns:
            List of search results
        """
        if self.conn is None or self.model is None:
            raise RuntimeError("DB or model not initialized")

        # Query embedding
        query_embedding = self.model.encode([query], is_query=True)[0]
        query_bytes = self._serialize_vector(query_embedding)

        # Vector search (sqlite-vec knn query)
        limit = top_k * 2 if domain_filter else top_k
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
            })

            if len(output) >= top_k:
                break

        return output

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build sqlite-vec vector database from chunked parquet files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="vectors.db",
        help="Database filename (default: vectors.db)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="Export Milvus/Qdrant compatible parquet file",
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

    args = parser.parse_args()

    if args.list_models:
        print("Supported embedding models:")
        print("-" * 50)
        for model, dim in MODEL_CONFIGS.items():
            default = " (default)" if model == DEFAULT_MODEL else ""
            print(f"  {model} (dim={dim}){default}")
        return 0

    db_path = args.output_dir / args.db_name
    builder = VectorDBBuilder(db_path, model_name=args.model)

    try:
        # Build
        stats = builder.build(args.input_dir)

        if "error" in stats:
            return 1

        # Export parquet
        if args.export_parquet:
            export_path = args.output_dir / "vectors_export.parquet"
            builder.export_for_milvus(export_path)

        # Test search
        if args.test_search:
            print(f"\n🔍 Test search: '{args.test_search}'")
            print("-" * 50)

            results = builder.search(args.test_search, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] Similarity: {r['similarity']:.4f}")
                print(f"    File: {r['source_file']}")
                print(f"    Section: {' > '.join(r['section_path'])}")
                print(f"    Content: {r['chunk_text'][:100]}...")

        return 0

    finally:
        builder.close()


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        exit(130)
