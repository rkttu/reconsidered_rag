# Reconsidered RAG

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

**[한국어](README.ko.md)** | English

> **Production RAG needs production data.**
>
> Before you build RAG, prepare your data pipeline.

---

## The Problem

Most RAG tools do this:

```text
Documents → [Black Box] → Vector DB
```

**Problems:**

- ❌ Can't inspect intermediate results
- ❌ Can't iterate without re-embedding ($$$)
- ❌ Can't version control the transformation
- ❌ Can't verify data quality before embedding

## Our Approach

```text
Documents → Markdown → Parquet → [Your Choice]
            ↑          ↑
         Inspect    Re-embeddable
         Edit       Version Control
```

**Benefits:**

- ✅ **Human-readable checkpoints** (Markdown) - Review and edit before embedding
- ✅ **Iterate before embedding** - Fix issues without paying twice
- ✅ **Version control with Git** - Track every transformation
- ✅ **Embedding model independence** - Re-embed anytime with any model

---

## Who Is This For?

### ✅ You Need This If:

**Scenario 1: Large-scale structured data**

```text
"I have a PostgreSQL database with 1,000 tables.
I need to experiment with which tables/columns to include."
```

→ Reconsidered RAG lets you iterate on extraction logic without re-running expensive embedding.

**Scenario 2: Data quality issues**

```text
"My PDFs have OCR errors and need manual review before RAG."
```

→ Reconsidered RAG gives you Markdown checkpoints you can review and edit.

**Scenario 3: Research reproducibility**

```text
"I'm comparing different chunking strategies for my paper.
I need to version control the entire pipeline."
```

→ Reconsidered RAG produces Git-friendly artifacts you can commit and reproduce.

**Scenario 4: Cost optimization**

```text
"I want to test locally first, then move to cloud when ready.
I don't want vendor lock-in."
```

→ Reconsidered RAG separates data preparation from embedding, giving you freedom to choose.

### ❌ You DON'T Need This If:

| You Want... | Use This Instead |
|------------|------------------|
| Quick prototype (5 minutes) | LangChain or LlamaIndex |
| Fully managed solution | Azure AI Search, Pinecone |
| No iteration needed | Your data is already perfect |

---

## Quick Start

```bash
# Install
uv sync

# Run (prepare + chunk in one command)
uv run python main.py run
```

**Output**: `chunked_data/*.parquet` - Text chunks ready for any embedding model.

<details>
<summary><b>With LLM Enrichment (Optional)</b></summary>

```bash
# Set environment variables
export ENRICHMENT_ENDPOINT="https://your-endpoint.openai.azure.com/"
export ENRICHMENT_API_KEY="your-api-key"
export ENRICHMENT_MODEL="gpt-4o"

# Run with enrichment
uv run python main.py run --enrich
```

</details>

<details>
<summary><b>Step-by-Step (Power Users)</b></summary>

```bash
# Step 1: Document → Markdown
uv run python main.py prepare --source all

# Step 2: LLM enrichment (optional)
uv run python main.py enrich

# Step 3: Markdown → Parquet chunks
uv run python main.py chunk
```

</details>

---

## The Workflow

### Phase 1: Data Preparation (Free, Iterative)

```text
┌─────────────────────────────────────────┐
│ 1. Extract & Transform (Free)          │
├─────────────────────────────────────────┤
│ 01_prepare_*.py → prepared_contents/    │
│                                         │
│ Output: Markdown files                  │
│ - Human-readable                        │
│ - Git-friendly                          │
│ - Editable                              │
└─────────────────────────────────────────┘
              ↓
       [Inspect & Iterate]
       - Review samples
       - Fix OCR errors
       - Add context
       - Remove noise
              ↓
┌─────────────────────────────────────────┐
│ 2. Chunk & Structure (Free)            │
├─────────────────────────────────────────┤
│ 03_chunk_content.py → chunked_data/     │
│                                         │
│ Output: Parquet files                   │
│ - Text chunks                           │
│ - Structure metadata                    │
│ - Re-embeddable                         │
└─────────────────────────────────────────┘
              ↓
        [Verify Chunks]
        - Check sizes
        - Review splits
        - Test samples
```

### Phase 2: Embedding ($$$ - Your Choice, One Time)

```text
┌─────────────────────────────────────────┐
│ 3. Choose Your Embedding Strategy      │
├─────────────────────────────────────────┤
│                                         │
│ Option A: Local (Free)                  │
│   → BGE-M3, E5, MiniLM                  │
│   → CPU/GPU, no API costs               │
│                                         │
│ Option B: Cloud (Paid)                  │
│   → OpenAI, Cohere, AWS Bedrock         │
│   → High quality, pay per token         │
│                                         │
│ Option C: Try Both!                     │
│   → Same Parquet, different embeddings  │
│   → Compare quality/cost                │
│                                         │
└─────────────────────────────────────────┘
```

**Key Point**: You only pay for embedding **after** your data is perfect.

---

## Pipeline Details

| Step | Script | Input | Output | Cost |
|------|--------|-------|--------|------|
| **1a** | `01_prepare_markdowndocs.py` | MD, TXT, RST | Markdown | Free |
| **1b** | `01_prepare_officedocs.py` | DOCX, PDF, PPTX, media | Markdown | Free |
| **2** | `02_enrich_content.py` | Markdown | Enriched MD | Optional (LLM) |
| **3** | `03_chunk_content.py` | Markdown | Parquet | Free |

<details>
<summary><b>Extensible: Add Your Own Data Sources</b></summary>

The `01_prepare_*` naming convention allows multiple data source handlers:

| Future Script | Data Source | Status |
|--------------|-------------|--------|
| `01_prepare_postgresql.py` | PostgreSQL database | 📝 Plan |
| `01_prepare_discourse.py` | Forum exports | 📝 Plan |
| `01_prepare_github.py` | GitHub Issues/PRs | 📝 Plan |
| `01_prepare_slack.py` | Slack exports | 📝 Plan |

All output Markdown → same enrichment → same chunking → your choice of embedding.

</details>

---

## Use the Parquet Anywhere

The core output is **Parquet files** containing text chunks and metadata. Use them with any embedding model or vector database:

```python
import pandas as pd

# Load chunks
df = pd.read_parquet("chunked_data/your_document.parquet")
texts = df["chunk_text"].tolist()
metadata = df[["heading_text", "section_path", "chunk_type"]]

# Then: Choose your embedding
# - OpenAI, Cohere, AWS Bedrock
# - Local: BGE-M3, E5, Sentence Transformers
# - Or no embedding: BM25, keyword search

# Then: Choose your vector DB
# - Pinecone, Qdrant, Milvus
# - Elasticsearch, OpenSearch
# - SQLite-vec (included example)
```

| Approach | Compatible With |
|----------|----------------|
| **Vector RAG** | Any embedding → Any vector DB |
| **BM25 / Keyword** | Elasticsearch, Typesense, Meilisearch |
| **Hybrid Search** | Vector + BM25 combined |
| **Reranking** | Cohere, BGE-Reranker |
| **Analytics** | DuckDB, Polars, pandas |

---

## Two Checkpoints, Two Opportunities

| Checkpoint | Format | Purpose | Actions |
|-----------|--------|---------|---------|
| **prepared_contents/** | Markdown | Human review | Fix OCR, add context, remove noise |
| **chunked_data/** | Parquet | Ready to embed | Test chunking, verify structure |

Both are **Git-friendly** - track changes, review diffs, collaborate.

---

## Comparison

### Think of it as:

```text
dbt: SQL → transformed SQL (for data warehouses)
Reconsidered RAG: Documents → transformed text (for RAG systems)
```

### vs. Other Tools

| Tool | Category | Focus | Reconsidered RAG |
|------|----------|-------|------------------|
| **LangChain** | RAG Framework | End-to-end RAG | Data preparation layer |
| **LlamaIndex** | RAG Framework | End-to-end RAG | Pre-RAG pipeline |
| **Azure AI Search** | Search Service | Managed search | Offline preparation |
| **Unstructured.io** | Document parsing | Text extraction | + Checkpoints + Git |

**Position**: Not a replacement, but a **preprocessing layer** you can use before any RAG tool.

---

## Case Studies

### Case 1: PostgreSQL Database (1,000 Tables)

**Problem**: Need to build RAG from large database, but don't know which tables/columns to include.

**Solution**:

```python
# 01_prepare_postgresql.py (you write this)
TABLE_CONFIGS = {
    'posts': {
        'columns': ['id', 'title', 'content'],
        'where': "published_at > '2023-01-01'",
    },
    'comments': {
        'columns': ['id', 'text'],
        'join': 'posts ON comments.post_id = posts.id',
    },
    # Explicitly include 10 tables, ignore 990
}
```

**Result**:

- ✅ Control dimension explosion (990 tables avoided)
- ✅ Iterate on table selection (free - just SQL queries)
- ✅ Embed only final selection ($$$)

**Savings**: 100x embedding cost reduction

### Case 2: 10,000 PDFs with OCR Errors

**Problem**: PDFs have OCR errors that will contaminate RAG results.

**Solution**:

```bash
# 1. Extract to Markdown
uv run python 01_prepare_officedocs.py

# 2. Random sample review
ls prepared_contents/ | shuf -n 10 | xargs cat

# 3. Found errors? Fix them
vim prepared_contents/problematic_doc.md

# Or improve extraction script
vim 01_prepare_officedocs.py
# Re-run (free - no embedding yet)

# 4. Git commit
git add prepared_contents/
git commit -m "Fix OCR errors in medical docs"

# 5. Now chunk and embed (once)
uv run python 03_chunk_content.py
```

**Savings**: Avoided re-embedding after discovering issues late

### Case 3: Research Paper (Chunking Comparison)

**Problem**: Comparing different chunking strategies, need reproducible results.

**Solution**:

```bash
# Prepare data once
uv run python 01_prepare_markdowndocs.py
git add prepared_contents/
git commit -m "Source documents prepared"

# Try different chunking
uv run python 03_chunk_content.py --max-chunk-size 500
git add chunked_data/
git tag experiment-chunk-500

uv run python 03_chunk_content.py --max-chunk-size 1000
git add chunked_data/
git tag experiment-chunk-1000

# Compare results
git diff experiment-chunk-500 experiment-chunk-1000
```

**Benefit**: Full pipeline reproducibility for academic papers

---

## Supported Formats

| Category | Extensions |
|----------|-----------|
| **Office** | DOCX, XLSX, PPTX |
| **PDF/Web** | PDF, HTML, JSON, CSV |
| **Text** | MD, TXT, RST |
| **Images** | JPG, PNG (OCR) |
| **Audio** | MP3, WAV (speech-to-text) |
| **Video** | MP4, MKV (subtitle extraction) |
| **Code** | PY, JS, TS, IPYNB, etc. |

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for full list and configuration.

---

## Optional: Local Vector DB + MCP Server

For testing and development, build a local vector database:

```bash
# Install with vector DB support
uv sync --extra vectordb --extra mcp

# Build vector DB + run MCP server
uv run python example_sqlitevec_mcp.py all
```

<details>
<summary><b>More Options</b></summary>

```bash
# Build with different embedding model
uv run python example_sqlitevec_mcp.py build --model intfloat/multilingual-e5-large

# Run MCP server (SSE mode for web clients)
uv run python example_sqlitevec_mcp.py serve --sse --port 8080

# List supported models
uv run python example_sqlitevec_mcp.py --list-models
```

**Available Models**:

- `BAAI/bge-m3` (default, 1024 dim, multilingual)
- `intfloat/multilingual-e5-large` (1024 dim)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast)

</details>

---

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Install uv

**Windows (PowerShell)**:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# Core only (no optional features)
uv sync

# With LLM enrichment
uv sync --extra enrich

# With vector DB
uv sync --extra vectordb

# Full pipeline
uv sync --extra all
```

---

## Documentation

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: Full installation guide, Docker setup, IDE integration
- **[HISTORY.md](HISTORY.md)**: Design decisions and evolution (why we made these choices)

---

## What's Different?

### vs. LangChain/LlamaIndex

| Aspect | LangChain/LlamaIndex | Reconsidered RAG |
|--------|---------------------|------------------|
| **Goal** | End-to-end RAG system | Data preparation pipeline |
| **Speed** | ⚡ Fast (5 minutes) | 🐌 Thorough (30 minutes) |
| **Control** | ⚠️ Automated | ✅ Manual checkpoints |
| **Iteration** | 💸 Re-embed each time | ✅ Free until satisfied |
| **Git-friendly** | ❌ No | ✅ Yes (Markdown + Parquet) |

**When to use what**:

- **Quick prototype**: LangChain/LlamaIndex
- **Production data pipeline**: Reconsidered RAG → then LangChain/LlamaIndex

### vs. Unstructured.io

| Aspect | Unstructured.io | Reconsidered RAG |
|--------|----------------|------------------|
| **Focus** | Document parsing | Parsing + pipeline + checkpoints |
| **Checkpoints** | ❌ No | ✅ Markdown + Parquet |
| **Pricing** | $$ API service | Free (open source) |
| **Git workflow** | ❌ No | ✅ Yes |

---

## Philosophy

### "Reconsidered" Means:

We questioned common RAG assumptions:

1. **"Faster is better"** → Reconsidered: Thorough is better for production data
2. **"Automate everything"** → Reconsidered: Humans should verify transformations
3. **"Embed immediately"** → Reconsidered: Embed only after data is perfect

See [HISTORY.md](HISTORY.md) for the full journey.

---

## Roadmap

### Current (v0.4.x)

- ✅ Document preparation (Office, PDF, media)
- ✅ LLM enrichment (optional)
- ✅ Structure-based chunking
- ✅ Example: sqlite-vec + MCP server

### Planned

- [ ] More data sources (PostgreSQL, Discourse, GitHub, Slack)
- [ ] More chunking strategies (sentence-level, sliding window)
- [ ] Direct export to vector DB formats (Qdrant JSON, Pinecone JSONL)
- [ ] Parallel processing for large datasets

### Won't Add

- ❌ Bundled embedding models (you choose)
- ❌ Specific vector DB integrations in core (keep it portable)
- ❌ LLM generation (we focus on the "R" in RAG)

---

## Contributing

Contributions welcome! See [Contributing Guide](#contributing) for details.

**Areas where help is needed**:

1. **Data source plugins**: PostgreSQL, MongoDB, APIs
2. **Documentation**: Tutorials, case studies
3. **Testing**: Edge cases, large-scale testing

---

## License

[Apache License 2.0](LICENSE)

---

## Support

- 📖 [Documentation](IMPLEMENTATION.md)
- 💬 [GitHub Discussions](https://github.com/rkttu/reconsidered_rag/discussions)
- 🐛 [Issue Tracker](https://github.com/rkttu/reconsidered_rag/issues)
- 💖 [GitHub Sponsors](https://github.com/sponsors/rkttu)

---

## Acknowledgments

Built with:

- [markitdown](https://github.com/microsoft/markitdown) - Microsoft's document conversion
- [pymupdf4llm](https://github.com/pymupdf/PyMuPDF4LLM) - PDF processing
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - Vector search in SQLite

Inspired by:

- [dbt](https://www.getdbt.com/) - Data transformation workflow
- [Unstructured.io](https://unstructured.io/) - Document parsing
- The principle of **reproducible research**

---

**Remember**: Production RAG needs production data. Prepare first, embed last.
