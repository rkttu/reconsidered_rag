# Reconsidered RAG

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

**[English](README.md)** | 한국어

**Any Data → Markdown → Parquet: RAG 준비 완료, 오프라인, 이식 가능.**

---

## 핵심 요약

> **빠른 RAG DB 빌더가 아닙니다.**
> **데이터를 소유하고 싶은 사람들을 위한 RAG 도구 상자입니다.**
>
> 빠른 RAG 구축을 원한다면 LangChain이나 LlamaIndex가 정답입니다.
> **데이터 주권**과 **벤더 종속 방지**를 원한다면 잘 오셨습니다.

---

## 누구를 위한 프로젝트?

| ❌ 맞지 않는 경우 | ✅ 맞는 경우 |
| --------------- | ----------- |
| 5분 안에 RAG 구축 원함 | 이식 가능한 포맷으로 **데이터 소유** 원함 |
| 벤더 종속 괜찮음 | **언제든 재임베딩** 가능해야 함 |
| 블랙박스 파이프라인 선호 | **사람이 읽을 수 있는 체크포인트** 필요 |

---

## 세 가지 사용 사례

| 💰 인프라 부족 | 🔐 데이터 주권 | ⚡ 빠른 시작 + 통제 |
| ------------- | ------------- | ----------------- |
| GPU 없음, 클라우드 없음 | 데이터가 내 머신을 떠나지 않음 | 2개 명령으로 시작 |
| ~200MB 설치 | 이식 가능한 포맷 (MD, Parquet) | 어떤 체크포인트든 편집 |
| 준비되면 나중에 비용 지불 | Git 친화적, 감사 가능 | 어떤 CSP, 어떤 모델이든 |

---

## 빠른 시작

```bash
uv sync
uv run python main.py run
```

끝. `chunked_data/*.parquet` 확인.

<details>
<summary><b>LLM 보강 포함</b></summary>

```bash
uv run python main.py run --enrich
```
</details>

<details>
<summary><b>단계별 실행 (고급)</b></summary>

```bash
uv run python main.py prepare   # 1. 문서 → Markdown
uv run python main.py enrich    # 2. LLM 보강 (선택)
uv run python main.py chunk     # 3. Markdown → Parquet
```
</details>

---

## 작동 방식

```text
모든 데이터 소스       →    MARKDOWN    →    PARQUET (텍스트만)
────────────────────────────────────────────────────────────────
파일 (PDF, DOCX...)        구조화됨        → 어떤 임베딩 모델이든
DB (PostgreSQL...)         사람이 읽음     → 어떤 벡터 DB든
API (GitHub, Slack...)     Git 친화적     → BM25, 하이브리드, 리랭킹
웹 (Discourse, Wiki...)    편집 가능      → 파인튜닝 데이터
```

**우리는 "R" (Retrieval-ready)을 제공. "AG" (Augmented Generation)는 당신이 결정.**

---

## 파이프라인

| 단계 | 스크립트 | 역할 |
| ---- | -------- | ---- |
| 1a | `01_prepare_markdowndocs.py` | MD/TXT/RST → Markdown |
| 1b | `01_prepare_officedocs.py` | Office/PDF/미디어 → Markdown |
| 2 | `02_enrich_content.py` | LLM 보강 (선택) |
| 3 | `03_chunk_content.py` | 구조 기반 청킹 → Parquet |

<details>
<summary><b>확장: 데이터 소스 추가</b></summary>

| 예정 스크립트 | 데이터 소스 |
| ------------ | ----------- |
| `01_prepare_discourse.py` | PostgreSQL 포럼 덤프 |
| `01_prepare_github.py` | GitHub Issues/PR |
| `01_prepare_slack.py` | Slack 내보내기 |
| `01_prepare_notion.py` | Notion API |
| `01_prepare_database.py` | 모든 SQL DB |

모두 Markdown 출력 → 동일한 보강 → 동일한 청킹.
</details>

---

## Parquet 활용

```python
import pandas as pd
df = pd.read_parquet("chunked_data/your_document.parquet")
texts = df["chunk_text"].tolist()

# 그 다음: OpenAI, Cohere, AWS Bedrock, 로컬 ONNX — 당신의 선택
# 그 다음: Pinecone, Qdrant, Milvus, Elasticsearch — 당신의 선택
```

| 접근 방식 | 호환 |
| -------- | ---- |
| 벡터 RAG | 어떤 임베딩 → 어떤 벡터 DB |
| BM25 / 키워드 | Elasticsearch, Typesense, Meilisearch |
| 하이브리드 검색 | 벡터 + BM25 결합 |
| 리랭킹 | Cohere, BGE-Reranker |
| 분석 | DuckDB, Polars |

---

## 두 개의 체크포인트

| `prepared_contents/` | `chunked_data/` |
| -------------------- | --------------- |
| 편집 가능한 Markdown | 이식 가능한 Parquet |
| OCR 오류 수정, 맥락 추가 | 텍스트 청크 + 구조 메타데이터 |
| Git 친화적 | 어떤 임베딩이든 준비됨 |

---

## 선택: 로컬 벡터 DB + MCP

```bash
uv sync --extra vectordb --extra mcp
uv run python example_sqlitevec_mcp.py all
```

<details>
<summary><b>추가 옵션</b></summary>

```bash
# 다른 모델로 빌드
uv run python example_sqlitevec_mcp.py build --model intfloat/multilingual-e5-large

# MCP 서버 실행 (SSE 모드)
uv run python example_sqlitevec_mcp.py serve --sse --port 8080
```
</details>

---

## 지원 포맷

**오피스**: DOCX, XLSX, PPTX | **PDF/웹**: PDF, HTML, JSON, CSV | **텍스트**: MD, TXT, RST
**이미지**: JPG, PNG (OCR) | **오디오**: MP3, WAV (STT) | **비디오**: MP4, MKV (자막) | **코드**: PY, JS, TS 등

---

## 문서

설치, 설정, Docker, IDE 연동은 **[IMPLEMENTATION.md](IMPLEMENTATION.md)** 참고.

---

## 라이선스 & 기여

[Apache License 2.0](LICENSE) | [기여 가이드](#기여)

[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

<details>
<summary><b>기여</b></summary>

1. Fork → 2. Branch → 3. Commit → 4. Push → 5. PR
</details>
