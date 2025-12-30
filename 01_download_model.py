"""
01_download_model.py
BGE-M3 Embedding Model Download Module

Downloads BAAI/bge-m3 model and caches it locally.
Multilingual embedding model for semantic chunking.

Features:
- Multilingual support (Korean, English, Chinese, etc. 100+ languages)
- Supports Dense, Sparse, ColBERT three embedding types
- 1024-dimension dense embeddings
"""

from pathlib import Path
from typing import Any
from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]


# Cache directory setup
CACHE_DIR = Path(__file__).parent / "cache" / "huggingface"

# Model configuration
MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"


def download_model() -> bool:
    """
    Download BGE-M3 model and reranker model

    Downloads models through FlagEmbedding library and
    saves them to local cache.

    Returns:
        bool: Success status
    """
    print(f"🔄 Downloading BGE-M3 model: {MODEL_NAME}")
    print("(First run downloads ~2.2GB)")
    print()

    try:
        # Load BGE-M3 model (automatically downloads)
        model: Any = BGEM3FlagModel(MODEL_NAME, use_fp16=True, cache_dir=str(CACHE_DIR))

        print()
        print("✅ BGE-M3 model download completed!")
        print()

        # Test
        print("🔍 Testing BGE-M3 embeddings...")
        test_texts = [
            "Hello, this is a semantic chunking test.",
            "Hello, this is a semantic chunking test."
        ]

        embeddings = model.encode(test_texts)
        dense_vecs = embeddings["dense_vecs"]

        print(f"   • Number of input texts: {len(test_texts)}")
        print(f"   • Embedding dimensions: {dense_vecs.shape}")
        print("✅ BGE-M3 test successful!")

        # Download BGE reranker model
        print()
        print(f"🔄 Downloading BGE reranker model: {RERANKER_MODEL_NAME}")
        print("(First run downloads ~1.1GB)")
        print()

        reranker: Any = BGEM3FlagModel(RERANKER_MODEL_NAME, use_fp16=False, cache_dir=str(CACHE_DIR))  # CPU mode

        print()
        print("✅ BGE reranker model download completed!")
        print()

        # Test reranker
        print("🔍 Testing reranker...")
        query = "semantic chunking"
        candidates = ["Semantic chunking is a technology that divides text into meaningful units.", "Vector search calculates similarity."]
        scores = reranker.compute_score([[query, cand] for cand in candidates])

        print(f"   • Query: {query}")
        print(f"   • Number of candidates: {len(candidates)}")
        print(f"   • Scores: {scores}")
        print("✅ Reranker test successful!")

        return True

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Main function"""
    print("=" * 50)
    print("BGE-M3 Embedding Model Download")
    print("=" * 50)
    print()

    success = download_model()

    if success:
        print()
        print("🎉 Model preparation completed!")
        print("   You can run semantic chunking with the following command:")
        print("   python 03_semantic_chunking.py")
        print("   Reranking feature will be activated in MCP server.")

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        exit(130)
