"""
embedding_model.py
PIXIE-Rune ONNX Embedding Model Module

Provides a unified embedding interface using ONNX Runtime.
Can be used by semantic chunking, vector DB building, and MCP server.
"""

from pathlib import Path
from typing import Optional
import numpy as np

# Directory setup
BASE_DIR = Path(__file__).parent
ONNX_MODEL_DIR = BASE_DIR / "cache" / "onnx_model"
CACHE_DIR = BASE_DIR / "cache" / "huggingface"

# Model configuration
MODEL_NAME = "telepix/PIXIE-Rune-Preview"
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 8192


def detect_device() -> tuple[str, list[str]]:
    """
    GPU/CPU 환경을 감지하여 적절한 디바이스와 ONNX Provider를 반환
    
    Returns:
        tuple: (pytorch_device, onnx_providers)
            - pytorch_device: "cuda" or "cpu"
            - onnx_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"] or ["CPUExecutionProvider"]
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except ImportError:
        pass
    
    return "cpu", ["CPUExecutionProvider"]


class PIXIEEmbeddingModel:
    """PIXIE-Rune ONNX Embedding Model"""
    
    def __init__(self, use_onnx: bool = True):
        """
        Initialize embedding model
        
        Args:
            use_onnx: Use ONNX model (True) or PyTorch model (False)
        """
        self.use_onnx = use_onnx
        self.model = None
        self.tokenizer = None
        self.ort_model = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Lazy initialization of the model"""
        if self._initialized:
            return
        
        if self.use_onnx and ONNX_MODEL_DIR.exists():
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        self._initialized = True
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model using optimum"""
        import sys
        import time
        
        def _log(msg: str) -> None:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [embedding] {msg}", file=sys.stderr, flush=True)
        
        _log("ONNX 모델 로딩 시작...")
        _log(f"경로: {ONNX_MODEL_DIR}")
        
        # GPU/CPU 감지
        pytorch_device, onnx_providers = detect_device()
        _log(f"디바이스 감지: {pytorch_device}, ONNX Providers: {onnx_providers}")
        
        _log("transformers import 중...")
        from transformers import AutoTokenizer
        _log("optimum import 중...")
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        _log("import 완료")
        
        _log("ORTModelForFeatureExtraction.from_pretrained() 호출 중...")
        load_start = time.time()
        self.ort_model = ORTModelForFeatureExtraction.from_pretrained(
            str(ONNX_MODEL_DIR),
            provider=onnx_providers[0],  # 주요 provider 지정
        )
        _log(f"ONNX 모델 로딩 완료 ({time.time() - load_start:.2f}s)")
        
        _log("AutoTokenizer.from_pretrained() 호출 중...")
        tok_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ONNX_MODEL_DIR),
            # fix_mistral_regex는 이 토크나이저와 호환되지 않음
        )
        _log(f"토크나이저 로딩 완료 ({time.time() - tok_start:.2f}s)")
        
        _log("ONNX 모델 초기화 완료")
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model using sentence-transformers"""
        from sentence_transformers import SentenceTransformer
        import sys
        import os
        
        os.environ["HF_HOME"] = str(CACHE_DIR)
        
        # GPU/CPU 감지
        pytorch_device, _ = detect_device()
        print(f"🔄 PIXIE-Rune PyTorch 모델 로딩 중... (device: {pytorch_device})", file=sys.stderr, flush=True)
        
        self.model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(CACHE_DIR),
            device=pytorch_device,  # 명시적으로 디바이스 지정
        )
        print(f"✅ PyTorch 모델 로딩 완료 (device: {self.model.device})", file=sys.stderr, flush=True)
    
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        is_query: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            is_query: Whether texts are queries (adds query prefix)
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of embeddings [N, 1024]
        """
        self.initialize()
        
        if not texts:
            return np.array([])
        
        if self.use_onnx and self.ort_model is not None:
            return self._encode_onnx(texts, batch_size, is_query, normalize)
        else:
            return self._encode_pytorch(texts, batch_size, is_query, normalize)
    
    def _encode_onnx(
        self,
        texts: list[str],
        batch_size: int,
        is_query: bool,
        normalize: bool,
    ) -> np.ndarray:
        """Encode using ONNX model"""
        if self.tokenizer is None or self.ort_model is None:
            raise RuntimeError("ONNX model not initialized")
        
        all_embeddings = []
        
        # Add query prefix if needed
        if is_query:
            texts = [f"query: {t}" for t in texts]
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            
            # Run ONNX inference
            outputs = self.ort_model(**inputs)
            
            # CLS pooling (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            if normalize:
                # L2 normalization
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def _encode_pytorch(
        self,
        texts: list[str],
        batch_size: int,
        is_query: bool,
        normalize: bool,
    ) -> np.ndarray:
        """Encode using PyTorch model (sentence-transformers)"""
        if self.model is None:
            raise RuntimeError("PyTorch model not initialized")
        
        if is_query:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                prompt_name="query",
                normalize_embeddings=normalize,
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )
        
        return embeddings.astype(np.float32)
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            
        Returns:
            Similarity matrix [N, M]
        """
        # Normalize (in case not already normalized)
        norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        
        return np.dot(norm1, norm2.T)


# Global singleton instance
_embedding_model: Optional[PIXIEEmbeddingModel] = None


def get_embedding_model(use_onnx: bool = True) -> PIXIEEmbeddingModel:
    """Get or create the global embedding model instance"""
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = PIXIEEmbeddingModel(use_onnx=use_onnx)
    
    return _embedding_model


def encode_texts(
    texts: list[str],
    batch_size: int = 32,
    is_query: bool = False,
) -> np.ndarray:
    """
    Convenience function to encode texts
    
    Args:
        texts: List of texts to encode
        batch_size: Batch size for encoding
        is_query: Whether texts are queries
        
    Returns:
        numpy array of embeddings
    """
    model = get_embedding_model()
    return model.encode(texts, batch_size=batch_size, is_query=is_query)


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("PIXIE-Rune Embedding Model Test")
    print("=" * 60)
    print()
    
    model = get_embedding_model(use_onnx=True)
    
    # Test queries
    queries = [
        "대한민국의 수도는 어디인가?",
        "What is semantic chunking?",
    ]
    
    # Test documents
    documents = [
        "대한민국의 수도는 서울이다.",
        "Semantic chunking divides text into meaningful units.",
        "Python is a programming language.",
    ]
    
    print("🔍 Testing embeddings...")
    query_emb = model.encode(queries, is_query=True)
    doc_emb = model.encode(documents, is_query=False)
    
    print(f"   Query embeddings: {query_emb.shape}")
    print(f"   Document embeddings: {doc_emb.shape}")
    
    # Compute similarity
    sim = model.compute_similarity(query_emb, doc_emb)
    print(f"\n📊 Similarity matrix:")
    print(sim)
    
    print("\n✅ Test completed!")
