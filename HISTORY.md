# Reconsidered: A History of Rethinking

This document records the evolution of design decisions in this project. The name "Reconsidered RAG" reflects our commitment to questioning assumptions and iterating toward better solutions.

---

## Timeline

### v0.1.0 — Original Philosophy (2024)

Initial Vision: "Responsible Refusal"

The project started with an ambitious philosophy:

> RAG systems should know when to refuse answering, not just retrieve and generate.

Key principles:

- Explicit edit points over implicit automation
- Delay meaning fixation (embedding) as late as possible
- Refusal is not failure
- Accountability must remain with humans

**Implementation:**

- PIXIE-Rune ONNX embedding model for CPU-only processing
- Semantic chunking based on embedding similarity
- Full vector DB pipeline included
- 5-step process: download model → prepare content → semantic chunking → build vector DB → MCP server

---

### v0.1.x — First Feedback (2025)

**Criticism Received:**

> "Too slow."
> "Who would compute embeddings twice?"

The "twice embedding" problem:

```text
Document → Markdown → [Embedding #1: for semantic chunking] → Parquet
                                    ↓
                         [Embedding #2: for vector DB]
```

**Gap Identified:**

The original philosophy of "responsible refusal" couldn't be implemented with embedding models alone. Actual refusal decisions require LLM reasoning, not just similarity scores.

| Philosophy | Implementation Reality |
| ------------ | ---------------------- |
| "Know when to refuse" | Embedding models can't refuse |
| "Delay meaning fixation" | I was computing embeddings twice |
| "Accountability" | No mechanism existed |

---

### v0.2.0 — The Pivot (January 2026)

Rediscovering the Original Intent

Before asking "what if I remove embedding?", I first asked: *What was this project really trying to do?*

The answer emerged through conversation:

> "The original intent was to create **portable RAG datasets** — Parquet files that can be migrated to any vector DB."

This wasn't about embedding models at all. It was about:

1. **Portability** — Prepare once, import anywhere
2. **Human-readable checkpoints** — Markdown that people can review and edit
3. **Delayed decisions** — Don't lock into a specific embedding model or vector DB

**The Two Checkpoint Design:**

I realized the pipeline had two valuable intervention points:

| Checkpoint | Format | Purpose |
| ------------ | -------- | --------- |
| `prepared_contents/` | Markdown | Human review, OCR correction, context addition |
| `chunked_data/` | Parquet | Portable text, re-embeddable with any model |

The Markdown checkpoint is particularly valuable:

- Fix OCR errors before they propagate
- Add context that automated extraction missed
- Remove noise and irrelevant sections
- Version control with Git

**Then Came the Question:**

With this clarity, I asked:

> What if I remove embedding entirely from this project?

**Analysis of Semantic Chunking:**

| Aspect | Semantic Chunking | Structure-based Chunking |
| -------- | ------------------ | ------------------------- |
| Embedding required | ✅ Yes | ❌ No |
| Speed | Slow | Fast |
| Retrieval quality | Slightly better (theory) | Good enough (practice) |
| Predictability | Low (threshold tuning) | High (heading/paragraph) |
| "Twice embedding" | Applies | Does not apply |

Research findings (2024-2025):

- LangChain benchmarks: No significant difference between semantic and fixed-size chunking
- LlamaIndex experiments: Fixed-size with overlap was more stable
- Modern embedding models (8K+ tokens) reduce chunking sensitivity

**The Context Window Revolution:**

Embedding models have evolved dramatically:

| Model | Max Tokens | Year |
| ------- | ----------- | ------ |
| OpenAI ada-002 | 8,191 | 2022 |
| Cohere embed-v3 | 512 | 2024 |
| Jina v3 | 8,192 | 2024 |
| Voyage-3 | 32,000 | 2025 |

With 8K+ token context windows becoming standard:

- **No need for aggressive chunking** — A well-structured Markdown section fits entirely
- **No need for overlap tricks** — Arbitrary overlap ratios (10%? 20%?) become irrelevant
- **No need for semantic similarity thresholds** — Heading/paragraph boundaries are clear and deterministic

The insight: **Well-structured Markdown documents, split at natural boundaries (headings, paragraphs), are sufficient for modern embedding models.**

I don't need complex chunking strategies. I need good document structure.

**Decision: Remove embedding entirely.**

---

## What Changed in v0.2.0

### Removed

- `01_download_model.py` — No embedding model needed
- `03_semantic_chunking.py` — Replaced with structure-based
- `embedding_model.py` — PIXIE-Rune wrapper removed
- Heavy dependencies: sentence-transformers, flagembedding, accelerate, optimum, etc.

### Added

- `02_chunk_content.py` — Structure-based chunking (no embedding)

### Renamed

- `02_prepare_content.py` → `01_prepare_content.py`
- `04_build_vector_db.py` → `03_build_vector_db.py`
- `05_mcp_server.py` → `05_mcp_server.py`

### New Pipeline

**Before (v0.1.0):** 5 steps, embedding required

```text
Document → Model Download → Markdown → Semantic Chunking → Vector DB → MCP
              (slow)                      (embedding)        (embedding)
```

**After (v0.2.0):** 2 core steps, embedding optional

```text
Document → Markdown → Structure Chunking → Parquet (text only)
                         (fast)              ↓
                                    [Your choice of embedding]
```

### Dependency Reduction

| Category | v0.1.0 | v0.2.0 |
| ---------- | -------- | -------- |
| Core dependencies | 20+ packages | 8 packages |
| Install size | ~2GB+ | ~200MB |
| GPU required | No (but slow) | No (and fast) |
| Embedding model | Bundled | Your choice |

---

## New Positioning

### What This Project Does

✅ Converts documents to Markdown (with optional OCR/enrichment)
✅ Chunks by structure (headings, paragraphs, tables)
✅ Exports as Parquet (text only, no embeddings)

### What This Project Does NOT Do

❌ Choose your embedding model
❌ Manage your vector database
❌ Serve production RAG

### Why This Is Better

| Benefit | Explanation |
| --------- | ------------- |
| **Model freedom** | Embedding models change fast; text in Parquet, re-embed anytime |
| **Speed** | No embedding computation, just text processing |
| **Simplicity** | 2 steps instead of 5 |
| **Portability** | Parquet works everywhere |
| **Auditability** | Markdown checkpoint for human review |

---

## Lessons Learned

### 1. Philosophy ≠ Implementation

Having a good philosophy (responsible refusal, accountability) is not enough. The implementation must support it. Embedding models cannot make refusal decisions.

### 2. Semantic Chunking Was Oversold

In 2022-2023, semantic chunking seemed promising. By 2025-2026, with 8K+ token embedding models and cross-encoder reranking, the difference became negligible.

### 3. Coupling Creates Problems

Bundling embedding model choice into a document preparation tool creates unnecessary coupling. Separation of concerns is better.

### 4. "Slow but correct" Still Needs Justification

Saying "it's slow but that's okay" doesn't work if the slowness doesn't provide proportional value.

---

## Future Considerations

### What I Might Add

- [ ] More chunking strategies (sentence-level, sliding window)
- [ ] Token counting for different models
- [ ] Direct export to vector DB formats (Qdrant JSON, Pinecone JSONL)
- [ ] Parallel processing for large document sets

### What I Will NOT Add

- ❌ Bundled embedding models
- ❌ Specific vector DB integrations in core
- ❌ LLM-based refusal logic (this was the original dream, but belongs in a different layer)

---

## v0.3.0 — Full Pipeline Restored (January 2026)

### The Flexibility Pivot

After removing embedding from v0.2.0, I asked:

> "What if I want to test the full pipeline locally?"

The strict "offline, no embedding" philosophy was too rigid. The original intent was **flexibility**, not dogma:

> "The intention was not to adhere to an offline philosophy, but to be flexible enough to adjust as circumstances dictate."

### What's New in v0.3.0

#### 1. LLM Enrichment (Optional)

`01_prepare_content.py` now supports `--enrich` flag:

```bash
# Set Microsoft Foundry / Azure OpenAI credentials
export ENRICHMENT_ENDPOINT="https://your-endpoint.inference.ai.azure.com"
export ENRICHMENT_API_KEY="your-api-key"
export ENRICHMENT_MODEL="gpt-4.1"

# Run with enrichment
uv run python 01_prepare_content.py --enrich
```

Enrichment adds to YAML front matter:

| Field | Description |
| ----- | ----------- |
| `llm_summary` | AI-generated 2-3 sentence summary |
| `llm_keywords` | Semantic keywords beyond headings |
| `llm_questions` | Questions this document can answer |
| `llm_entities` | Named entities (tools, frameworks, concepts) |
| `llm_difficulty` | beginner / intermediate / advanced |

#### 2. Local Embedding with BGE-M3/E5

`03_build_vector_db.py` now uses local models via sentence-transformers:

```bash
# Default: BAAI/bge-m3 (multilingual, 1024 dim)
uv run python 03_build_vector_db.py

# Alternative models
uv run python 03_build_vector_db.py --model intfloat/multilingual-e5-large
uv run python 03_build_vector_db.py --model sentence-transformers/all-MiniLM-L6-v2

# List available models
uv run python 03_build_vector_db.py --list-models
```

Supported models:

| Model | Dimension | Best For |
| ----- | --------- | -------- |
| `BAAI/bge-m3` | 1024 | General multilingual |
| `intfloat/multilingual-e5-large` | 1024 | Cross-lingual retrieval |
| `intfloat/multilingual-e5-base` | 768 | Balanced size/quality |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast English-only |
| `sentence-transformers/all-mpnet-base-v2` | 768 | High quality English |

#### 3. MCP Server Restored

`05_mcp_server.py` provides a complete RAG testing environment:

```bash
# stdio mode (for Claude Desktop, etc.)
uv run python 05_mcp_server.py

# SSE mode (for HTTP clients)
uv run python 05_mcp_server.py --sse --port 8080
```

Available tools:

| Tool | Description |
| ---- | ----------- |
| `search` | Vector similarity search with natural language |
| `get_chunk` | Retrieve full chunk by ID |
| `list_documents` | List all indexed documents |
| `get_stats` | Database statistics |

### Architecture v0.3.0

```text
                        ┌─────────────────────────────────────┐
                        │         Core Pipeline (v0.2.0)      │
                        │                                     │
Documents ──┬──────────▶│ 01_prepare_content.py ──────────▶  │──▶ Markdown
            │           │                                     │    (prepared_contents/)
            │           │ 02_chunk_content.py ────────────▶   │──▶ Parquet
            │           │                                     │    (chunked_data/)
            │           └─────────────────────────────────────┘
            │
            │                 Optional Extensions (v0.3.0)
            │           ┌─────────────────────────────────────┐
            ├──[enrich]▶│ Microsoft Foundry GPT-4.1           │
            │           │ • llm_summary, llm_keywords         │
            │           │ • llm_questions, llm_entities       │
            │           └─────────────────────────────────────┘
            │
            │           ┌─────────────────────────────────────┐
            ├─[vectordb]│ 03_build_vector_db.py               │──▶ sqlite-vec
            │           │ • BGE-M3, E5, MiniLM models         │    (vector_db/)
            │           └─────────────────────────────────────┘
            │
            │           ┌─────────────────────────────────────┐
            └───[mcp]──▶│ 05_mcp_server.py                    │──▶ MCP Protocol
                        │ • search, get_chunk, list_documents │    (stdio/SSE)
                        └─────────────────────────────────────┘
```

### Dependency Changes

| Extra | Packages |
| ----- | -------- |
| `enrich` | azure-ai-inference, openai |
| `vectordb` | sqlite-vec, sentence-transformers, numpy |
| `mcp` | mcp, sqlite-vec, sentence-transformers, starlette, uvicorn |
| `all` | All of the above |

```bash
# Core only (no optional features)
uv sync

# With enrichment
uv sync --extra enrich

# With vector DB
uv sync --extra vectordb

# Full pipeline
uv sync --extra all
```

---

## v0.3.1 — 5-Step Pipeline Restructure (January 2026)

### Explicit Step Separation

After implementing v0.3.0 with `--enrich` as an option in step 1, I reconsidered the pipeline structure:

> "What if I want to run enrichment separately, or skip it when API keys aren't set?"

The flag-based approach had limitations:

| Problem | Impact |
| ------- | ------ |
| Mixed concerns | Document conversion + LLM enrichment in same step |
| All-or-nothing | Can't re-enrich without re-processing |
| Silent failure | Easy to forget the flag |

### What's New in v0.3.1

#### Explicit 5-Step Pipeline

```text
01_prepare_content.py   → Document → Markdown (pure conversion)
02_enrich_content.py    → LLM enrichment (auto-skip if no .env)
03_chunk_content.py     → Structure-based chunking
04_build_vector_db.py   → BGE-M3/E5 local embedding
05_mcp_server.py        → MCP server for testing
```

#### Auto-Skip Behavior

Step 2 (enrichment) now checks for environment variables:

```bash
# If these are not set, step 2 prints a message and exits cleanly
ENRICHMENT_ENDPOINT="https://your-endpoint.inference.ai.azure.com"
ENRICHMENT_API_KEY="your-api-key"
```

#### Re-Enrichment Support

```bash
# Re-enrich all documents (even already enriched ones)
uv run python 02_enrich_content.py --force
```

### Benefits of Separation

| Aspect | v0.3.0 (`--enrich` flag) | v0.3.1 (separate step) |
| ------ | ------------------------ | ---------------------- |
| Re-processing | Must redo step 1 | Only run step 2 |
| Batch control | All or nothing | Per-step granularity |
| CI/CD | Flag management | Step skipping |
| Debugging | Interleaved logs | Clean separation |

### CLI Changes

```bash
# v0.3.0 (removed)
uv run python 01_prepare_content.py --enrich  # ❌ No longer available

# v0.3.1 (new)
uv run python 02_enrich_content.py            # ✅ Separate step
uv run python 02_enrich_content.py --force    # ✅ Re-enrich all
```

### The Lesson

Flexibility > Purity

v0.2.0 was pure but limiting. v0.3.0 keeps the core pure while offering practical extensions:

| Component | Philosophy |
| --------- | ---------- |
| Core (01, 02) | Offline, no API required |
| Enrich | Optional LLM enhancement |
| VectorDB | Optional local embedding |
| MCP | Optional testing server |

The core pipeline remains:

```bash
uv run python 01_prepare_content.py  # No API needed
uv run python 02_chunk_content.py    # No API needed
```

Everything else is opt-in.

---

## References

- [LangChain Chunking Benchmarks](https://blog.langchain.dev/)
- [LlamaIndex Chunking Experiments](https://docs.llamaindex.ai/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [BGE-M3 Paper](https://arxiv.org/abs/2402.03216)

---

## The Name "Reconsidered"

The name remains relevant:

| Version | What I Reconsidered |
| --------- | --------------------- |
| v0.1.0 | RAG itself — do I need GPU? cloud? |
| v0.2.0 | Our own assumptions — do I need bundled embeddings? |

**Reconsidering is not failure. It's the process.**

---

## v0.4.0 — Core vs Example Separation (January 2026)

### The Clarity Pivot

After implementing the 5-step pipeline in v0.3.1, I reconsidered the structure:

> "Steps 1-3 produce the high-quality Parquet file. That's the core value.
> Steps 4-5 (vector DB + MCP server) are just one way to USE that Parquet."

The numbered steps implied a linear dependency:

```text
01 → 02 → 03 → 04 → 05  (suggests all 5 are required)
```

But the reality was:

```text
01 → 02 → 03 → Parquet (CORE)
                   ↓
            [Your choice]
            ├── sqlite-vec + MCP
            ├── Chroma
            ├── Pinecone
            └── ...
```

### What Changed in v0.4.0

#### Merged and Renamed

| Before (v0.3.1) | After (v0.4.0) |
| --------------- | -------------- |
| `04_build_vector_db.py` | `example_sqlitevec_mcp.py` |
| `05_mcp_server.py` | (merged into above) |

#### New File Structure

```text
01_prepare_content.py    ─┐
02_enrich_content.py      ├─▶ Core Pipeline (produces Parquet)
03_chunk_content.py      ─┘

example_sqlitevec_mcp.py ◀─ Application Example (one of many possible)
```

#### Usage Changes

```bash
# Before (v0.3.1)
uv run python 04_build_vector_db.py
uv run python 05_mcp_server.py

# After (v0.4.0)
uv run python example_sqlitevec_mcp.py build
uv run python example_sqlitevec_mcp.py serve
uv run python example_sqlitevec_mcp.py all  # build + serve
```

### Benefits of Separation

| Aspect | Before | After |
| ------ | ------ | ----- |
| Clarity | "5 required steps" | "3 core + examples" |
| Extensibility | Add step 06, 07? | Add example_chroma.py, etc. |
| Naming | Implies sequence | Implies choice |
| Mental model | Pipeline | Core + Plugins |

### The Lesson

**Naming matters for user understanding.**

Numbered files (01, 02, 03, 04, 05) imply a mandatory sequence.
Naming something `example_*` signals it's optional and replaceable.

The core pipeline:
- **01_prepare_content.py** — Document → Markdown
- **02_enrich_content.py** — LLM enrichment (auto-skip if no API key)
- **03_chunk_content.py** — Structure-based chunking → Parquet

Everything else is an example of what you can DO with that Parquet.

---

## v0.4.1 — Modular Prepare Scripts (January 2026)

### The Specialization Pivot

After separating core pipeline from examples in v0.4.0, I reconsidered Step 1:

> "What if I want to handle Discourse PostgreSQL dumps? Or GitHub Issues?
> The `01_prepare_content.py` that handles Office docs shouldn't be the same
> script that parses SQL dumps."

### What Changed in v0.4.1

#### Split 01_prepare_content.py

| Before | After |
| ------ | ----- |
| `01_prepare_content.py` | `prepare_utils.py` (shared utilities) |
| (monolithic) | `01_prepare_markdowndocs.py` (MD, TXT, RST) |
| | `01_prepare_officedocs.py` (DOCX, XLSX, PPTX, PDF, media) |

#### New File Structure

```text
prepare_utils.py              ← Common: metadata extraction, YAML generation

01_prepare_markdowndocs.py    ← Text-based: pass-through + metadata
01_prepare_officedocs.py      ← Binary formats: markitdown/pymupdf4llm
01_prepare_discourse.py       ← (future) PostgreSQL forum dump
01_prepare_github.py          ← (future) GitHub API
01_prepare_slack.py           ← (future) Slack export
```

#### Naming Convention

All `01_prepare_*` scripts:
- Accept data from different sources
- Output to `prepared_contents/` as Markdown
- Share `prepare_utils.py` for metadata extraction

### Benefits of Separation

| Aspect | Monolithic | Modular |
| ------ | ---------- | ------- |
| SQL dump handling | Bloat in main script | Separate `01_prepare_discourse.py` |
| Dependency isolation | All deps always loaded | Only needed deps per script |
| Testing | One giant test | Focused unit tests |
| Dimension explosion | Risk of processing unwanted tables | Filter at prepare stage |

### The Discourse Example

For a PostgreSQL forum dump:

```bash
# Future script would:
# 1. Connect to SQL dump
# 2. SELECT posts + topics (ignore notifications, user_actions)
# 3. Join and format as Markdown per topic
# 4. Output to prepared_contents/

uv run python 01_prepare_discourse.py --dump forum.sql
```

The key insight: **Data selection and filtering happens at Step 1**,
not later in the pipeline. This prevents dimension explosion.

---

## v0.4.2 — Clear Positioning (January 2026)

### The Identity Pivot

After all the restructuring, I asked the fundamental question:

> "Who is this project actually for?"

The answer crystallized:

> **This is NOT a fast RAG DB builder.**
> **This is a tool for people who want to own their data.**

### What Changed in v0.4.2

#### 1. Clear Positioning in README

| Before | After |
| ------ | ----- |
| "RAG-ready document preparation" | "NOT a fast RAG builder" |
| Feature-focused | User-focused |
| Implicit value | Explicit trade-offs |

#### 2. Fast Path CLI

New `main.py` with single command:

```bash
# Before (4 commands)
uv run python 01_prepare_markdowndocs.py
uv run python 01_prepare_officedocs.py
uv run python 03_chunk_content.py
uv run python example_sqlitevec_mcp.py all

# After (1 command)
uv run python main.py run
```

Options:
- `--source markdown|office|all` — Choose input type
- `--enrich` — Enable LLM enrichment
- `--input-dir` — Custom input directory

#### 3. Who Is This For?

| If you want... | This project is... |
| -------------- | ------------------ |
| Quick RAG in 5 minutes | ❌ Not for you |
| Lock-in to specific embedding | ❌ Not for you |
| Black-box pipeline | ❌ Not for you |
| **Own your data in portable formats** | ✅ For you |
| **Human-readable checkpoints** | ✅ For you |
| **Re-embed anytime with any model** | ✅ For you |
| **Migrate to any vector DB** | ✅ For you |

### The Two Value Propositions

#### 💰 "Poor Man's RAG"

- No GPU required
- No cloud subscription required
- No vendor lock-in
- Start free, scale when ready

#### 🔐 "Data Sovereignty"

- Markdown: Human-readable, Git-versioned
- Parquet: Portable, vendor-agnostic
- No data leaves your machine (unless you choose)

### The Lesson

**Clarity > Features**

Having many features but unclear positioning confuses users.
Being honest about what you're NOT helps users self-select.

The project name "Reconsidered RAG" now makes more sense:
We're asking users to reconsider the trade-offs of convenience vs ownership.

---

Last updated: January 30, 2026
