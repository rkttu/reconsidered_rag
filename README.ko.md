# Reconsidered RAG

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

**[English](README.md)** | 한국어

> **프로덕션 RAG에는 프로덕션 데이터가 필요합니다.**
>
> RAG를 구축하기 전에, 데이터 파이프라인을 먼저 준비하세요.

---

## 문제 상황

대부분의 RAG 도구는 이렇게 작동합니다:

```text
문서 → [블랙박스] → 벡터 DB
```

**문제점:**

- ❌ 중간 결과를 검사할 수 없음
- ❌ 재임베딩 없이 반복 작업 불가능 ($$$)
- ❌ 변환 과정의 버전 관리 불가능
- ❌ 임베딩 전 데이터 품질 검증 불가능

## 우리의 접근 방식

```text
문서 → Markdown → Parquet → [당신의 선택]
       ↑          ↑
    검사 가능   재임베딩 가능
    편집 가능   버전 관리 가능
```

**장점:**

- ✅ **사람이 읽을 수 있는 체크포인트** (Markdown) - 임베딩 전 검토 및 편집
- ✅ **임베딩 전에 반복 작업** - 두 번 비용 내지 않고 문제 수정
- ✅ **Git으로 버전 관리** - 모든 변환 과정 추적
- ✅ **임베딩 모델 독립성** - 언제든 어떤 모델로든 재임베딩 가능

---

## 누구를 위한 프로젝트인가?

### ✅ 이런 경우 필요합니다:

**시나리오 1: 대규모 구조화 데이터**

```text
"1,000개 테이블이 있는 PostgreSQL 데이터베이스가 있습니다.
어떤 테이블/컬럼을 포함할지 실험해야 합니다."
```

→ Reconsidered RAG는 비싼 임베딩을 재실행하지 않고 추출 로직을 반복 수정할 수 있게 해줍니다.

**시나리오 2: 데이터 품질 문제**

```text
"PDF에 OCR 오류가 있어서 RAG 전에 수동 검토가 필요합니다."
```

→ Reconsidered RAG는 검토하고 편집할 수 있는 Markdown 체크포인트를 제공합니다.

**시나리오 3: 연구 재현성**

```text
"논문을 위해 서로 다른 청킹 전략을 비교하고 있습니다.
전체 파이프라인의 버전 관리가 필요합니다."
```

→ Reconsidered RAG는 커밋하고 재현할 수 있는 Git 친화적 산출물을 생성합니다.

**시나리오 4: 비용 최적화**

```text
"먼저 로컬에서 테스트하고, 준비되면 클라우드로 이동하고 싶습니다.
벤더 종속을 원하지 않습니다."
```

→ Reconsidered RAG는 데이터 준비를 임베딩과 분리해서 선택의 자유를 줍니다.

### ❌ 이런 경우에는 필요 없습니다:

| 원하는 것... | 대신 이것을 사용 |
|------------|-----------------|
| 빠른 프로토타입 (5분) | LangChain 또는 LlamaIndex |
| 완전 관리형 솔루션 | Azure AI Search, Pinecone |
| 반복 작업이 필요 없음 | 데이터가 이미 완벽함 |

---

## 빠른 시작

```bash
# 설치
uv sync

# 실행 (prepare + chunk 한 번에)
uv run python main.py run
```

**출력**: `chunked_data/*.parquet` - 어떤 임베딩 모델로도 사용 가능한 텍스트 청크.

<details>
<summary><b>LLM 보강 포함 (선택사항)</b></summary>

```bash
# 환경 변수 설정
export ENRICHMENT_ENDPOINT="https://your-endpoint.openai.azure.com/"
export ENRICHMENT_API_KEY="your-api-key"
export ENRICHMENT_MODEL="gpt-4o"

# 보강 포함 실행
uv run python main.py run --enrich
```

</details>

<details>
<summary><b>단계별 실행 (고급 사용자)</b></summary>

```bash
# 1단계: 문서 → Markdown
uv run python main.py prepare --source all

# 2단계: LLM 보강 (선택)
uv run python main.py enrich

# 3단계: Markdown → Parquet 청크
uv run python main.py chunk
```

</details>

---

## 워크플로우

### 1단계: 데이터 준비 (무료, 반복 가능)

```text
┌─────────────────────────────────────────┐
│ 1. 추출 & 변환 (무료)                   │
├─────────────────────────────────────────┤
│ 01_prepare_*.py → prepared_contents/    │
│                                         │
│ 출력: Markdown 파일                      │
│ - 사람이 읽을 수 있음                    │
│ - Git 친화적                            │
│ - 편집 가능                             │
└─────────────────────────────────────────┘
              ↓
       [검사 & 반복]
       - 샘플 검토
       - OCR 오류 수정
       - 맥락 추가
       - 노이즈 제거
              ↓
┌─────────────────────────────────────────┐
│ 2. 청킹 & 구조화 (무료)                 │
├─────────────────────────────────────────┤
│ 03_chunk_content.py → chunked_data/     │
│                                         │
│ 출력: Parquet 파일                       │
│ - 텍스트 청크                            │
│ - 구조 메타데이터                        │
│ - 재임베딩 가능                          │
└─────────────────────────────────────────┘
              ↓
        [청크 검증]
        - 크기 확인
        - 분할 검토
        - 샘플 테스트
```

### 2단계: 임베딩 ($$$ - 당신의 선택, 한 번만)

```text
┌─────────────────────────────────────────┐
│ 3. 임베딩 전략 선택                     │
├─────────────────────────────────────────┤
│                                         │
│ 옵션 A: 로컬 (무료)                      │
│   → BGE-M3, E5, MiniLM                  │
│   → CPU/GPU, API 비용 없음               │
│                                         │
│ 옵션 B: 클라우드 (유료)                  │
│   → OpenAI, Cohere, AWS Bedrock         │
│   → 고품질, 토큰당 과금                  │
│                                         │
│ 옵션 C: 둘 다 시도!                      │
│   → 같은 Parquet, 다른 임베딩            │
│   → 품질/비용 비교                       │
│                                         │
└─────────────────────────────────────────┘
```

**핵심 포인트**: 데이터가 완벽해진 **후에만** 임베딩 비용을 지불합니다.

---

## 파이프라인 상세

| 단계 | 스크립트 | 입력 | 출력 | 비용 |
|------|---------|-----|------|------|
| **1a** | `01_prepare_markdowndocs.py` | MD, TXT, RST | Markdown | 무료 |
| **1b** | `01_prepare_officedocs.py` | DOCX, PDF, PPTX, 미디어 | Markdown | 무료 |
| **2** | `02_enrich_content.py` | Markdown | 보강된 MD | 선택 (LLM) |
| **3** | `03_chunk_content.py` | Markdown | Parquet | 무료 |

<details>
<summary><b>확장 가능: 자체 데이터 소스 추가</b></summary>

`01_prepare_*` 네이밍 규칙으로 여러 데이터 소스 핸들러 추가 가능:

| 예정 스크립트 | 데이터 소스 | 상태 |
|-------------|------------|------|
| `01_prepare_postgresql.py` | PostgreSQL 데이터베이스 | 📝 계획 |
| `01_prepare_discourse.py` | 포럼 내보내기 | 📝 계획 |
| `01_prepare_github.py` | GitHub Issues/PR | 📝 계획 |
| `01_prepare_slack.py` | Slack 내보내기 | 📝 계획 |

모든 출력 Markdown → 같은 보강 → 같은 청킹 → 당신의 임베딩 선택.

</details>

---

## Parquet 어디서든 사용

핵심 출력은 텍스트 청크와 메타데이터를 포함한 **Parquet 파일**입니다. 어떤 임베딩 모델이나 벡터 데이터베이스와도 사용 가능:

```python
import pandas as pd

# 청크 로드
df = pd.read_parquet("chunked_data/your_document.parquet")
texts = df["chunk_text"].tolist()
metadata = df[["heading_text", "section_path", "chunk_type"]]

# 그 다음: 임베딩 선택
# - OpenAI, Cohere, AWS Bedrock
# - 로컬: BGE-M3, E5, Sentence Transformers
# - 또는 임베딩 없이: BM25, 키워드 검색

# 그 다음: 벡터 DB 선택
# - Pinecone, Qdrant, Milvus
# - Elasticsearch, OpenSearch
# - SQLite-vec (예제 포함)
```

| 접근 방식 | 호환 |
|----------|-----|
| **벡터 RAG** | 어떤 임베딩 → 어떤 벡터 DB |
| **BM25 / 키워드** | Elasticsearch, Typesense, Meilisearch |
| **하이브리드 검색** | 벡터 + BM25 결합 |
| **리랭킹** | Cohere, BGE-Reranker |
| **분석** | DuckDB, Polars, pandas |

---

## 두 개의 체크포인트, 두 번의 기회

| 체크포인트 | 포맷 | 목적 | 할 수 있는 것 |
|-----------|-----|------|-------------|
| **prepared_contents/** | Markdown | 사람 검토 | OCR 수정, 맥락 추가, 노이즈 제거 |
| **chunked_data/** | Parquet | 임베딩 준비 | 청킹 테스트, 구조 검증 |

둘 다 **Git 친화적** - 변경 추적, diff 검토, 협업 가능.

---

## 비교

### 이렇게 생각하세요:

```text
dbt: SQL → 변환된 SQL (데이터 웨어하우스용)
Reconsidered RAG: 문서 → 변환된 텍스트 (RAG 시스템용)
```

### 다른 도구와 비교

| 도구 | 카테고리 | 초점 | Reconsidered RAG |
|-----|---------|-----|------------------|
| **LangChain** | RAG 프레임워크 | 엔드투엔드 RAG | 데이터 준비 레이어 |
| **LlamaIndex** | RAG 프레임워크 | 엔드투엔드 RAG | Pre-RAG 파이프라인 |
| **Azure AI Search** | 검색 서비스 | 관리형 검색 | 오프라인 준비 |
| **Unstructured.io** | 문서 파싱 | 텍스트 추출 | + 체크포인트 + Git |

**포지션**: 대체가 아닌, 모든 RAG 도구 **이전에** 사용하는 **전처리 레이어**.

---

## 사례 연구

### 사례 1: PostgreSQL 데이터베이스 (1,000개 테이블)

**문제**: 대규모 데이터베이스에서 RAG를 구축해야 하는데, 어떤 테이블/컬럼을 포함해야 할지 모름.

**해결책**:

```python
# 01_prepare_postgresql.py (직접 작성)
TABLE_CONFIGS = {
    'posts': {
        'columns': ['id', 'title', 'content'],
        'where': "published_at > '2023-01-01'",
    },
    'comments': {
        'columns': ['id', 'text'],
        'join': 'posts ON comments.post_id = posts.id',
    },
    # 10개 테이블만 명시적으로 포함, 990개 무시
}
```

**결과**:

- ✅ 차원 폭발 제어 (990개 테이블 제외)
- ✅ 테이블 선택 반복 수정 (무료 - SQL 쿼리만)
- ✅ 최종 선택만 임베딩 ($$$)

**절감**: 임베딩 비용 100배 감소

### 사례 2: OCR 오류가 있는 10,000개 PDF

**문제**: PDF에 OCR 오류가 있어서 RAG 결과를 오염시킬 것.

**해결책**:

```bash
# 1. Markdown으로 추출
uv run python 01_prepare_officedocs.py

# 2. 랜덤 샘플 검토
ls prepared_contents/ | shuf -n 10 | xargs cat

# 3. 오류 발견? 수정
vim prepared_contents/problematic_doc.md

# 또는 추출 스크립트 개선
vim 01_prepare_officedocs.py
# 재실행 (무료 - 아직 임베딩 전)

# 4. Git 커밋
git add prepared_contents/
git commit -m "의료 문서 OCR 오류 수정"

# 5. 이제 청킹과 임베딩 (한 번만)
uv run python 03_chunk_content.py
```

**절감**: 늦게 문제 발견 후 재임베딩 비용 방지

### 사례 3: 연구 논문 (청킹 비교)

**문제**: 서로 다른 청킹 전략을 비교하는데, 재현 가능한 결과 필요.

**해결책**:

```bash
# 데이터 한 번 준비
uv run python 01_prepare_markdowndocs.py
git add prepared_contents/
git commit -m "소스 문서 준비 완료"

# 다른 청킹 시도
uv run python 03_chunk_content.py --max-chunk-size 500
git add chunked_data/
git tag experiment-chunk-500

uv run python 03_chunk_content.py --max-chunk-size 1000
git add chunked_data/
git tag experiment-chunk-1000

# 결과 비교
git diff experiment-chunk-500 experiment-chunk-1000
```

**장점**: 학술 논문을 위한 완전한 파이프라인 재현성

---

## 지원 포맷

| 카테고리 | 확장자 |
|---------|-------|
| **오피스** | DOCX, XLSX, PPTX |
| **PDF/웹** | PDF, HTML, JSON, CSV |
| **텍스트** | MD, TXT, RST |
| **이미지** | JPG, PNG (OCR) |
| **오디오** | MP3, WAV (음성 인식) |
| **비디오** | MP4, MKV (자막 추출) |
| **코드** | PY, JS, TS, IPYNB 등 |

전체 목록과 설정은 [IMPLEMENTATION.md](IMPLEMENTATION.md) 참고.

---

## 선택사항: 로컬 벡터 DB + MCP 서버

테스트와 개발을 위해 로컬 벡터 데이터베이스 구축:

```bash
# 벡터 DB 지원 설치
uv sync --extra vectordb --extra mcp

# 벡터 DB 구축 + MCP 서버 실행
uv run python example_sqlitevec_mcp.py all
```

<details>
<summary><b>추가 옵션</b></summary>

```bash
# 다른 임베딩 모델로 구축
uv run python example_sqlitevec_mcp.py build --model intfloat/multilingual-e5-large

# MCP 서버 실행 (웹 클라이언트용 SSE 모드)
uv run python example_sqlitevec_mcp.py serve --sse --port 8080

# 지원 모델 목록
uv run python example_sqlitevec_mcp.py --list-models
```

**사용 가능 모델**:

- `BAAI/bge-m3` (기본값, 1024 차원, 다국어)
- `intfloat/multilingual-e5-large` (1024 차원)
- `sentence-transformers/all-MiniLM-L6-v2` (384 차원, 빠름)

</details>

---

## 설치

### 사전 요구사항

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) 패키지 매니저

### uv 설치

**Windows (PowerShell)**:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 의존성 설치

```bash
# 코어만 (선택 기능 없음)
uv sync

# LLM 보강 포함
uv sync --extra enrich

# 벡터 DB 포함
uv sync --extra vectordb

# 전체 파이프라인
uv sync --extra all
```

---

## 문서

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: 전체 설치 가이드, Docker 설정, IDE 연동
- **[HISTORY.md](HISTORY.md)**: 설계 결정과 발전 과정 (왜 이런 선택을 했는지)

---

## 무엇이 다른가?

### LangChain/LlamaIndex 대비

| 관점 | LangChain/LlamaIndex | Reconsidered RAG |
|-----|---------------------|------------------|
| **목표** | 엔드투엔드 RAG 시스템 | 데이터 준비 파이프라인 |
| **속도** | ⚡ 빠름 (5분) | 🐌 철저함 (30분) |
| **통제** | ⚠️ 자동화 | ✅ 수동 체크포인트 |
| **반복 작업** | 💸 매번 재임베딩 | ✅ 만족할 때까지 무료 |
| **Git 친화적** | ❌ 아니오 | ✅ 예 (Markdown + Parquet) |

**언제 무엇을**:

- **빠른 프로토타입**: LangChain/LlamaIndex
- **프로덕션 데이터 파이프라인**: Reconsidered RAG → 그 다음 LangChain/LlamaIndex

### Unstructured.io 대비

| 관점 | Unstructured.io | Reconsidered RAG |
|-----|----------------|------------------|
| **초점** | 문서 파싱 | 파싱 + 파이프라인 + 체크포인트 |
| **체크포인트** | ❌ 없음 | ✅ Markdown + Parquet |
| **가격** | $$ API 서비스 | 무료 (오픈소스) |
| **Git 워크플로우** | ❌ 없음 | ✅ 있음 |

---

## 철학

### "Reconsidered"의 의미:

우리는 일반적인 RAG 가정들에 의문을 제기했습니다:

1. **"빠를수록 좋다"** → 재고: 프로덕션 데이터에는 철저함이 더 좋다
2. **"모든 것을 자동화하라"** → 재고: 사람이 변환을 검증해야 한다
3. **"즉시 임베딩하라"** → 재고: 데이터가 완벽해진 후에만 임베딩하라

전체 여정은 [HISTORY.md](HISTORY.md) 참고.

---

## 로드맵

### 현재 (v0.4.x)

- ✅ 문서 준비 (오피스, PDF, 미디어)
- ✅ LLM 보강 (선택사항)
- ✅ 구조 기반 청킹
- ✅ 예제: sqlite-vec + MCP 서버

### 계획됨

- [ ] 더 많은 데이터 소스 (PostgreSQL, Discourse, GitHub, Slack)
- [ ] 더 많은 청킹 전략 (문장 단위, 슬라이딩 윈도우)
- [ ] 벡터 DB 포맷 직접 내보내기 (Qdrant JSON, Pinecone JSONL)
- [ ] 대규모 데이터셋 병렬 처리

### 추가하지 않을 것

- ❌ 번들 임베딩 모델 (당신이 선택)
- ❌ 코어에 특정 벡터 DB 통합 (이식성 유지)
- ❌ LLM 생성 (우리는 RAG의 "R"에 집중)

---

## 기여

기여를 환영합니다! 자세한 내용은 [기여 가이드](#기여) 참고.

**도움이 필요한 영역**:

1. **데이터 소스 플러그인**: PostgreSQL, MongoDB, API
2. **문서**: 튜토리얼, 사례 연구
3. **테스팅**: 엣지 케이스, 대규모 테스트

---

## 라이선스

[Apache License 2.0](LICENSE)

---

## 지원

- 📖 [문서](IMPLEMENTATION.md)
- 💬 [GitHub Discussions](https://github.com/rkttu/reconsidered_rag/discussions)
- 🐛 [이슈 트래커](https://github.com/rkttu/reconsidered_rag/issues)
- 💖 [GitHub Sponsors](https://github.com/sponsors/rkttu)

---

## 감사

Built with:

- [markitdown](https://github.com/microsoft/markitdown) - Microsoft의 문서 변환
- [pymupdf4llm](https://github.com/pymupdf/PyMuPDF4LLM) - PDF 처리
- [sentence-transformers](https://www.sbert.net/) - 임베딩 모델
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - SQLite 벡터 검색

Inspired by:

- [dbt](https://www.getdbt.com/) - 데이터 변환 워크플로우
- [Unstructured.io](https://unstructured.io/) - 문서 파싱
- **재현 가능한 연구**의 원칙

---

**기억하세요**: 프로덕션 RAG에는 프로덕션 데이터가 필요합니다. 먼저 준비하고, 마지막에 임베딩하세요.
