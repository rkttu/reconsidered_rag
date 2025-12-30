"""
01_download_model.py
BGE-M3 임베딩 모델 다운로드 모듈

BAAI/bge-m3 모델을 다운로드하여 로컬에 캐시합니다.
시맨틱 청킹에 사용할 다국어 임베딩 모델입니다.

특징:
- 다국어 지원 (한국어, 영어, 중국어 등 100+ 언어)
- Dense, Sparse, ColBERT 세 가지 임베딩 지원
- 1024 차원 밀집 임베딩
"""

from pathlib import Path
from typing import Any
from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]


# 캐시 디렉터리 설정
CACHE_DIR = Path(__file__).parent / "cache" / "huggingface"

# 모델 설정
MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"


def download_model() -> bool:
    """
    BGE-M3 모델 및 리랭커 모델 다운로드
    
    FlagEmbedding 라이브러리를 통해 모델을 다운로드하고
    로컬 캐시에 저장합니다.
    
    Returns:
        bool: 성공 여부
    """
    print(f"🔄 BGE-M3 모델 다운로드 중: {MODEL_NAME}")
    print("(처음 실행 시 약 2.2GB 다운로드)")
    print()
    
    try:
        # BGE-M3 모델 로드 (자동으로 다운로드됨)
        model: Any = BGEM3FlagModel(MODEL_NAME, use_fp16=True, cache_dir=str(CACHE_DIR))
        
        print()
        print("✅ BGE-M3 모델 다운로드 완료!")
        print()
        
        # 테스트
        print("🔍 BGE-M3 임베딩 테스트 중...")
        test_texts = [
            "안녕하세요, 시맨틱 청킹 테스트입니다.",
            "Hello, this is a semantic chunking test."
        ]
        
        embeddings = model.encode(test_texts)
        dense_vecs = embeddings["dense_vecs"]
        
        print(f"   • 입력 텍스트 수: {len(test_texts)}")
        print(f"   • 임베딩 차원: {dense_vecs.shape}")
        print("✅ BGE-M3 테스트 성공!")
        
        # BGE 리랭커 모델 다운로드
        print()
        print(f"🔄 BGE 리랭커 모델 다운로드 중: {RERANKER_MODEL_NAME}")
        print("(처음 실행 시 약 1.1GB 다운로드)")
        print()
        
        reranker: Any = BGEM3FlagModel(RERANKER_MODEL_NAME, use_fp16=False, cache_dir=str(CACHE_DIR))  # CPU 모드
        
        print()
        print("✅ BGE 리랭커 모델 다운로드 완료!")
        print()
        
        # 리랭커 테스트
        print("🔍 리랭커 테스트 중...")
        query = "시맨틱 청킹"
        candidates = ["시맨틱 청킹은 텍스트를 의미 단위로 나누는 기술입니다.", "벡터 검색은 유사도를 계산합니다."]
        scores = reranker.compute_score([[query, cand] for cand in candidates])
        
        print(f"   • 쿼리: {query}")
        print(f"   • 후보 수: {len(candidates)}")
        print(f"   • 점수: {scores}")
        print("✅ 리랭커 테스트 성공!")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """메인 함수"""
    print("=" * 50)
    print("BGE-M3 임베딩 모델 다운로드")
    print("=" * 50)
    print()
    
    success = download_model()
    
    if success:
        print()
        print("🎉 모델 준비 완료!")
        print("   다음 명령으로 시맨틱 청킹을 실행할 수 있습니다:")
        print("   python 03_semantic_chunking.py")
        print("   MCP 서버에서 리랭킹 기능이 활성화됩니다.")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[중단됨]")
        exit(130)
