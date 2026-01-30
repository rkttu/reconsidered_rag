# Reconsidered RAG

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

**[English](README.md)** | 한국어

[![데모 보기](https://img.youtube.com/vi/Uj6Vz5CZ4c4/maxresdefault.jpg)](https://youtu.be/Uj6Vz5CZ4c4)

**RAG를 위한 문서 준비: 오프라인, 이식 가능, 인프라 독립적.**

---

## 이 프로젝트가 하는 일

```mermaid
flowchart LR
    subgraph this["이 프로젝트"]
        A[문서<br/>PDF, DOCX, ...] --> B[Markdown<br/>+ 메타데이터]
        B --> C[구조 기반<br/>청킹]
        C --> D[Parquet<br/>텍스트만]
    end
    
    subgraph yours["당신의 선택"]
        E[OpenAI Embeddings]
        F[Cohere Embed]
        G[로컬 ONNX 모델]
        H[기타 임베딩 API]
    end
    
    subgraph vectordb["벡터 DB"]
        I[Pinecone]
        J[Qdrant]
        K[Milvus]
        L[Chroma]
    end
    
    D --> E
    D --> F
    D --> G
    D --> H
    E --> I
    F --> J
    G --> K
    H --> L
```

**이 프로젝트가 하는 일:**

- ✅ 문서를 Markdown으로 변환
- ✅ 구조 기반 청킹 (헤딩, 문단)
- ✅ Parquet으로 내보내기 (텍스트만)

**당신이 할 일:**

- 임베딩 모델 선택
- 벡터 DB 선택
- 프로덕션 서빙

---

## 왜 이 방식인가?

| 문제 | 우리의 해결책 |
| ---- | ------------ |
| 임베딩 모델이 빠르게 바뀜 | 텍스트가 Parquet에 있으니 언제든 재임베딩 |
| 벡터 DB를 결정 못함 | 한 번 준비하고 어디든 가져오기 |
| 데이터가 로컬을 떠날 수 없음 | 모든 것이 오프라인 실행 |
| 콘텐츠 감사/검토 필요 | 사람이 읽을 수 있는 Markdown 체크포인트 |

---

## 파이프라인

| 단계 | 스크립트 | 입력 | 출력 |
| ---- | -------- | ---- | ---- |
| 1 | `01_prepare_content.py` | 문서 (input_docs/) | Markdown (prepared_contents/) |
| 2 | `02_chunk_content.py` | Markdown | 청크 Parquet (chunked_data/) |

**끝. 2단계.**

임베딩 모델 없음. 벡터 DB 없음. 문서 준비만.

---

## 두 개의 사람이 읽을 수 있는 체크포인트

### 1. `prepared_contents/` — 편집 가능한 Markdown

- **자동 보강**: OCR, 이미지 설명, 음성-텍스트 (선택적, Azure AI 사용)
- **사람이 편집 가능**: 오류 수정, 맥락 추가, 노이즈 제거
- **버전 관리 가능**: 일반 텍스트는 Git과 호환

### 2. `chunked_data/` — 이식 가능한 Parquet

- **청크 텍스트 보존**: 어떤 모델로든 임베딩할 원본 텍스트
- **구조 정보**: `section_path`, `heading_level`, `element_type`
- **테이블 메타데이터**: `table_headers`, `table_row_count`

---

## 빠른 시작

```bash
# 의존성 설치 (최소)
uv sync

# 1. 문서 준비 (input_docs/에 파일 넣기)
uv run python 01_prepare_content.py

# 2. 구조 기반 청킹
uv run python 02_chunk_content.py

# 완료! chunked_data/*.parquet 확인
```

### Parquet 파일 사용하기

```python
import pandas as pd

# 청크 로드
df = pd.read_parquet("chunked_data/your_document.parquet")

# 임베딩할 텍스트 추출
texts = df["chunk_text"].tolist()

# 원하는 임베딩 모델 사용
from openai import OpenAI
client = OpenAI()
embeddings = client.embeddings.create(
    model="text-embedding-3-large",
    input=texts
).data

# 원하는 벡터 DB에 삽입
# ... 여기에 코드 작성
```

---

## 지원 파일 포맷

| 카테고리 | 확장자 |
| -------- | ------ |
| 오피스 | `.docx`, `.xlsx`, `.pptx` 등 |
| PDF/웹 | `.pdf`, `.html`, `.xml`, `.json`, `.csv` |
| Markdown/텍스트 | `.md`, `.txt`, `.rst` |
| 이미지 (EXIF/OCR) | `.jpg`, `.png`, `.webp` 등 |
| 오디오 (음성-텍스트) | `.mp3`, `.wav`, `.m4a` 등 |
| 비디오 (자막 추출) | `.mp4`, `.mkv`, `.avi` 등 |
| 코드 | `.py`, `.js`, `.ts`, `.java` 등 |

---

## 청킹 전략

**구조 기반 청킹**은 문서 구조를 존중합니다:

1. **헤딩 경계**: 각 헤딩이 새 청크를 시작
2. **테이블/코드/리스트**: 가능하면 그대로 유지
3. **큰 문단**: 문장 경계에서 오버랩과 함께 분할
4. **크기 설정 가능**: `--max-chunk-size`, `--min-chunk-size`

```bash
# 커스텀 청크 크기
uv run python 02_chunk_content.py --max-chunk-size 1500 --min-chunk-size 50
```

---

## 선택 사항: 벡터 DB & MCP 서버

sqlite-vec로 로컬 테스트를 원한다면:

```bash
# 선택적 의존성 설치
uv sync --extra vectordb
uv sync --extra mcp

# 벡터 DB 빌드 (임베딩 모델 필요)
uv run python 03_build_vector_db.py

# 테스트용 MCP 서버 실행
uv run python 05_mcp_server.py
```

---

## 상세 문서

설치, 설정, Docker, IDE 연동 등에 대해서는 **[IMPLEMENTATION.md](IMPLEMENTATION.md)**를 참고하세요.

---

## 라이선스

[Apache License 2.0](LICENSE)

## 후원

이 프로젝트가 도움이 되셨다면, GitHub Sponsors에서 후원을 고려해 주세요.

[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

## 기여하기

1. 이 저장소를 포크하세요
2. 브랜치를 생성하세요: `git checkout -b feature/amazing-feature`
3. 커밋하세요: `git commit -m 'Add amazing feature'`
4. 푸시하세요: `git push origin feature/amazing-feature`
5. Pull Request를 생성하세요
