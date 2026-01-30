# Reconsidered RAG

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

English | **[한국어](README.ko.md)**

**Any Data → Markdown → Parquet: RAG-ready, offline, portable.**

---

## TL;DR

> **This is NOT a fast RAG DB builder.**
> **This is a RAG toolbox for people who want to own their data.**
>
> Want quick RAG? LangChain or LlamaIndex is your answer.
> Want **data sovereignty** and **no vendor lock-in**? You're in the right place.

---

## Who Is This For?

| ❌ Not for you if... | ✅ For you if... |
| -------------------- | ---------------- |
| You want RAG in 5 minutes | You want to **own your data** in portable formats |
| You're okay with vendor lock-in | You want to **re-embed anytime** with any model |
| You prefer black-box pipelines | You need **human-readable checkpoints** |

---

## Three Use Cases

| 💰 Low Budget | 🔐 Data Sovereignty | ⚡ Fast + Controlled |
| ------------- | ------------------- | -------------------- |
| No GPU, no cloud | Data never leaves your machine | 2 commands to start |
| ~200MB install | Portable formats (MD, Parquet) | Edit any checkpoint |
| Pay later when ready | Git-friendly, auditable | Any CSP, any model |

---

## Quick Start

```bash
uv sync
uv run python main.py run
```

Done. Check `chunked_data/*.parquet`.

<details>
<summary><b>With LLM Enrichment</b></summary>

```bash
uv run python main.py run --enrich
```
</details>

<details>
<summary><b>Step-by-Step (Power Users)</b></summary>

```bash
uv run python main.py prepare   # 1. Documents → Markdown
uv run python main.py enrich    # 2. LLM enrichment (optional)
uv run python main.py chunk     # 3. Markdown → Parquet
```
</details>

---

## How It Works

```text
ANY DATA SOURCE        →    MARKDOWN    →    PARQUET (text only)
────────────────────────────────────────────────────────────────
Files (PDF, DOCX...)        Structured       → Any embedding model
Databases (PostgreSQL...)   Human-readable   → Any vector DB
APIs (GitHub, Slack...)     Git-friendly     → BM25, hybrid, rerank
Web (Discourse, Wiki...)    Editable         → Fine-tuning data
```

**We provide "R" (Retrieval-ready data). You decide "AG" (Augmented Generation).**

---

## Pipeline

| Step | Script | What it does |
| ---- | ------ | ------------ |
| 1a | `01_prepare_markdowndocs.py` | MD/TXT/RST → Markdown |
| 1b | `01_prepare_officedocs.py` | Office/PDF/Media → Markdown |
| 2 | `02_enrich_content.py` | LLM enrichment (optional) |
| 3 | `03_chunk_content.py` | Structure-based chunking → Parquet |

<details>
<summary><b>Extensible: Add your own data sources</b></summary>

| Future Script | Data Source |
| ------------- | ----------- |
| `01_prepare_discourse.py` | PostgreSQL forum dump |
| `01_prepare_github.py` | GitHub Issues/PRs |
| `01_prepare_slack.py` | Slack export |
| `01_prepare_notion.py` | Notion API |
| `01_prepare_database.py` | Any SQL database |

All output Markdown → same enrichment → same chunking.
</details>

---

## Use the Parquet Anywhere

```python
import pandas as pd
df = pd.read_parquet("chunked_data/your_document.parquet")
texts = df["chunk_text"].tolist()

# Then: OpenAI, Cohere, AWS Bedrock, local ONNX — your choice
# Then: Pinecone, Qdrant, Milvus, Elasticsearch — your choice
```

| Approach | Works with |
| -------- | ---------- |
| Vector RAG | Any embedding → Any vector DB |
| BM25 / Keyword | Elasticsearch, Typesense, Meilisearch |
| Hybrid Search | Vector + BM25 combined |
| Reranking | Cohere, BGE-Reranker |
| Analytics | DuckDB, Polars |

---

## Two Checkpoints

| `prepared_contents/` | `chunked_data/` |
| -------------------- | --------------- |
| Editable Markdown | Portable Parquet |
| Fix OCR errors, add context | Text chunks + structure metadata |
| Git-friendly | Ready for any embedding |

---

## Optional: Local Vector DB + MCP

```bash
uv sync --extra vectordb --extra mcp
uv run python example_sqlitevec_mcp.py all
```

<details>
<summary><b>More options</b></summary>

```bash
# Build with different model
uv run python example_sqlitevec_mcp.py build --model intfloat/multilingual-e5-large

# Run MCP server (SSE mode)
uv run python example_sqlitevec_mcp.py serve --sse --port 8080
```
</details>

---

## Supported Formats

**Office**: DOCX, XLSX, PPTX | **PDF/Web**: PDF, HTML, JSON, CSV | **Text**: MD, TXT, RST
**Images**: JPG, PNG (OCR) | **Audio**: MP3, WAV (STT) | **Video**: MP4, MKV (subtitles) | **Code**: PY, JS, TS, etc.

---

## Documentation

See **[IMPLEMENTATION.md](IMPLEMENTATION.md)** for installation, configuration, Docker, and IDE integration.

---

## License & Contributing

[Apache License 2.0](LICENSE) | [Contributing Guide](#contributing)

[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

<details>
<summary><b>Contributing</b></summary>

1. Fork → 2. Branch → 3. Commit → 4. Push → 5. PR
</details>
