"""
01_download_model.py
PIXIE-Rune Embedding Model Download & ONNX Conversion Module

Downloads telepix/PIXIE-Rune-Preview model and converts it to ONNX format.
Korean/English optimized embedding model for semantic chunking.

Features:
- Optimized for Korean and English
- 1024-dimension dense embeddings
- Max sequence length: 8192 tokens
- ONNX conversion for faster inference
"""

from pathlib import Path
import shutil


# Cache directory setup
CACHE_DIR = Path(__file__).parent / "cache" / "huggingface"
ONNX_MODEL_DIR = Path(__file__).parent / "cache" / "onnx_model"

# Model configuration
MODEL_NAME = "telepix/PIXIE-Rune-Preview"


def download_and_convert_model() -> bool:
    """
    Download PIXIE-Rune-Preview model and convert to ONNX

    Downloads model through sentence-transformers and converts
    to ONNX format using optimum library.

    Returns:
        bool: Success status
    """
    print(f"🔄 Downloading PIXIE-Rune-Preview model: {MODEL_NAME}")
    print("(First run downloads ~2.3GB)")
    print()

    try:
        # Step 1: Download model using sentence-transformers
        from sentence_transformers import SentenceTransformer
        import torch
        import os

        os.environ["HF_HOME"] = str(CACHE_DIR)

        print("📥 Loading model with sentence-transformers...")
        model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))

        print()
        print("✅ PIXIE-Rune-Preview model download completed!")
        print()

        # Step 2: Test the model
        print("🔍 Testing PIXIE-Rune-Preview embeddings...")
        test_queries = [
            "대한민국의 수도는 어디인가?",
            "What is semantic chunking?"
        ]
        test_docs = [
            "대한민국의 수도는 서울이다.",
            "Semantic chunking divides text into meaningful units."
        ]

        # Encode with query prompt
        query_embeddings = model.encode(test_queries, prompt_name="query")
        doc_embeddings = model.encode(test_docs)

        print(f"   • Query embeddings shape: {query_embeddings.shape}")
        print(f"   • Document embeddings shape: {doc_embeddings.shape}")

        # Calculate similarity
        similarities = model.similarity(query_embeddings, doc_embeddings)
        print(f"   • Similarity matrix:\n{similarities}")
        print("✅ PIXIE-Rune-Preview test successful!")
        print()

        # Step 3: Convert to ONNX
        print("🔄 Converting model to ONNX format...")
        print("(This may take a few minutes)")
        print()

        # Get the underlying transformer model path
        from transformers import AutoTokenizer, AutoModel
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        # Create ONNX output directory
        ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Export to ONNX using optimum
        print("📦 Exporting to ONNX...")

        # Load and export the model
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            MODEL_NAME,
            export=True,
            cache_dir=str(CACHE_DIR)
        )

        # Save the ONNX model
        ort_model.save_pretrained(str(ONNX_MODEL_DIR))

        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
        tokenizer.save_pretrained(str(ONNX_MODEL_DIR))

        print()
        print(f"✅ ONNX model saved to: {ONNX_MODEL_DIR}")
        print()

        # Step 4: Verify ONNX model
        print("🔍 Verifying ONNX model...")

        # Load and test ONNX model
        ort_model_loaded = ORTModelForFeatureExtraction.from_pretrained(str(ONNX_MODEL_DIR))
        tokenizer_loaded = AutoTokenizer.from_pretrained(str(ONNX_MODEL_DIR))

        # Test inference
        test_text = "ONNX 모델 테스트입니다."
        inputs = tokenizer_loaded(test_text, return_tensors="pt")
        outputs = ort_model_loaded(**inputs)

        print(f"   • Input text: {test_text}")
        print(f"   • Output shape: {outputs.last_hidden_state.shape}")
        print("✅ ONNX model verification successful!")

        # List ONNX files
        print()
        print("📁 ONNX model files:")
        for f in ONNX_MODEL_DIR.iterdir():
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"   • {f.name}: {size:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Main function"""
    print("=" * 60)
    print("PIXIE-Rune-Preview Model Download & ONNX Conversion")
    print("=" * 60)
    print()

    success = download_and_convert_model()

    if success:
        print()
        print("🎉 Model preparation completed!")
        print(f"   ONNX model location: {ONNX_MODEL_DIR}")
        print()
        print("   You can run semantic chunking with the following command:")
        print("   python 03_semantic_chunking.py")

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        exit(130)
