# Implementation Details

This document describes what the current reference implementation *happens to include*, not a recommended checklist for RAG systems.

Details may change as experiments evolve, but the principles and boundaries described in the [README](README.md) should not.

---

## Who Is This For?

> **This is NOT a fast RAG DB builder.**
> **This is a tool for people who want to own their data.**

| If you want... | This project is... |
| -------------- | ------------------ |
| Quick RAG in 5 minutes | ❌ Not for you |
| Lock-in to specific embedding | ❌ Not for you |
| **Own your data in portable formats** | ✅ For you |
| **Re-embed anytime with any model** | ✅ For you |

---

## Architecture

```mermaid
flowchart TD
    subgraph "Input"
        A[input_docs/<br/>Various format documents]
    end
    
    subgraph "Core Pipeline"
        B[01_prepare_*.py<br/>Document to Markdown]
        C[02_enrich_content.py<br/>LLM enrichment]
        D[03_chunk_content.py<br/>Structure-based chunking]
    end
    
    subgraph "Output"
        E[chunked_data/<br/>Parquet files]
    end
    
    subgraph "Application Examples"
        F[example_sqlitevec_mcp.py<br/>sqlite-vec + MCP server]
        G[example_chroma.py<br/>ChromaDB - future]
        H[example_pinecone.py<br/>Pinecone - future]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#fff3e0
    style F fill:#e8f5e9
    style G fill:#e0f2f1
    style H fill:#e0f2f1
```

Intermediate results from each step are saved as Markdown or Parquet files, making various experiments and external system migrations easy.

---

## Why PIXIE-Rune?

This project originally used the BGE-M3 model but switched to [PIXIE-Rune](https://huggingface.co/telepix/PIXIE-Rune-Preview) for **better Korean language performance**. PIXIE-Rune is optimized for Korean text understanding while maintaining strong multilingual capabilities.

Key differences from BGE-M3:

| Feature | BGE-M3 | PIXIE-Rune |
| ------- | ------ | ---------- |
| Korean Performance | Good | **Excellent** |
| Embedding Dimension | 1024 | 1024 |
| Max Sequence Length | 8192 | 8192 |
| ONNX Support | Built-in | **Manual conversion required** |

---

## Features

> [!NOTE]
> The following features reflect implementation choices made for this experiment.
They should not be interpreted as a recommended or complete RAG feature set.

- **Support for Various Document Formats**: Using Microsoft markitdown
  - Office documents: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
  - PDF, HTML, XML, JSON, CSV
  - Images (EXIF/OCR), Audio (speech recognition), Video (subtitle extraction)
  - Code files, Jupyter Notebook, ZIP archives
- **Microsoft Foundry Service Integration** (Optional)
  - Document Intelligence: Enhanced OCR for scanned PDFs and images
  - Azure OpenAI (GPT-4o): Image content understanding
  - Only configured services are automatically activated
- **Semantic Chunking**: Chunk splitting based on semantic similarity
- **Markdown Structure Preservation**: Maintains hierarchical information like heading levels and section paths
- **Enhanced Korean Support**: PIXIE-Rune model optimized for Korean text
- **Multilingual Support**: Strong performance across multiple languages
- **ONNX Optimization**: Pre-converted ONNX model for faster CPU inference
- **Incremental Updates**: Change detection based on content hash
- **zstd Compression**: Efficient parquet storage
- **BGE Reranker**: Reranking support for improved search result accuracy
- **CPU-Friendly**: Works without GPU (automatically uses GPU if available)

---

## Supported File Formats

| Category | Extensions |
| -------- | ---------- |
| Office Documents | `.docx`, `.doc`, `.xlsx`, `.xls`, `.pptx`, `.ppt` |
| PDF/Web | `.pdf`, `.html`, `.htm`, `.xml`, `.json`, `.csv` |
| Markdown/Text | `.md`, `.markdown`, `.txt`, `.rst` |
| Images (EXIF/OCR) | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff` |
| Audio (Speech Recognition) | `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` |
| Video (Subtitle Extraction) | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm` |
| Code/Other | `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.ipynb`, `.zip` |

---

## Module Structure

| Module | Description |
| ------ | ----------- |
| `prepare_utils.py` | Common utilities for all prepare scripts |
| `01_prepare_markdowndocs.py` | Markdown, TXT, RST → Markdown with metadata |
| `01_prepare_officedocs.py` | Office, PDF, media → Markdown with metadata |
| `02_enrich_content.py` | LLM-based content enrichment (optional) |
| `03_chunk_content.py` | Structure-based chunking to Parquet |
| `example_sqlitevec_mcp.py` | Application example: sqlite-vec + MCP server |

### Extensible Prepare Scripts

The `01_prepare_*` naming convention allows multiple data source handlers:

| Script | Data Source | Status |
| ------ | ----------- | ------ |
| `01_prepare_markdowndocs.py` | Markdown, TXT, RST | ✅ Available |
| `01_prepare_officedocs.py` | DOCX, XLSX, PPTX, PDF, images, audio, video | ✅ Available |
| `01_prepare_discourse.py` | PostgreSQL forum dump | 🔮 Future |
| `01_prepare_github.py` | GitHub Issues/PRs via API | 🔮 Future |
| `01_prepare_slack.py` | Slack export | 🔮 Future |

---

## Installation

### Installing uv

uv is a fast Python package manager. Install it from the official repository:

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or install via pip:

```bash
pip install uv
```

For more installation options, visit: <https://github.com/astral-sh/uv>.

### Installing Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install FlagEmbedding mistune pyarrow pandas pyyaml markitdown[all]
```

> **Note**: The `huggingface-hub[hf_xet]` package is included to improve download speeds for Xet Storage-supported models.

---

## Containerization

The program can be containerized using Docker. Model files are mounted as host volumes for caching.

### Build and Run

```bash
# Build image
docker-compose build

# Run (automatic model download → data processing → server start)
docker-compose up

# Or stdio mode
docker-compose run --rm aipack ./entrypoint.sh

# Specify port
PORT=9090 docker-compose up
```

### Execution Flow

The container automatically performs the following steps on startup:

1. **Model Download**: Check cache and download PIXIE-Rune embedding model + ONNX conversion
2. **Data Processing**: If `input_docs/` exists, prepare documents → chunking → vector DB build
3. **Server Start**: Run MCP server (SSE mode by default)

> **Note**: Works even in environments without uv or Python runtime (multi-stage build)

### Volume Mounts

- **Model Cache**: `./cache/huggingface` → `/root/.cache/huggingface` in container
  - Stores large files like BGE-M3 and reranker models in project's `cache/` directory
  - Explicit `cache_dir` setting in code controls cache location
  - Cache reuse on container restart saves download time
- **Data Directories**: `input_docs`, `prepared_contents`, `chunked_data`, `vector_db`
  - Data sharing between host and container

### Environment Variables

- `PYTHONUNBUFFERED=1`: Immediate log output

---

## Microsoft Foundry Service Integration (Optional)

Works with basic markitdown alone, but better results can be achieved by integrating Microsoft Foundry services.

### Supported Services

| Service | Purpose | Enhanced Features |
| ------- | ------- | ----------------- |
| Document Intelligence | Scanned PDF, image OCR | Text extraction accuracy |
| Azure OpenAI (GPT-4o) | Image content understanding | Image description generation |

### Configuration

```bash
# 1. Create environment file
cp .env.example .env

# 2. Enter only necessary keys (only configured services are activated)
```

Example `.env` file:

```env
# Document Intelligence (scanned PDF, image OCR)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Azure OpenAI (image content understanding)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

### Operation

- **No Keys**: Uses basic markitdown only
- **Some Services Configured**: Only those services are activated
- **All Configured**: Full functionality activated

Integration status is displayed during execution:

```text
🔗 Azure services integrated: Document Intelligence, OpenAI (gpt-4o)
```

Or:

```text
ℹ️ Azure services not integrated (using basic markitdown)
```

---

## Usage

### Fast Path (Recommended)

```bash
# Install and run in one step
uv sync
uv run python main.py run
```

Output: `chunked_data/*.parquet` — text chunks ready for any embedding model.

### With LLM Enrichment

```bash
uv run python main.py run --enrich
```

### Step-by-Step (Power Users)

#### 1. Document Preparation

Place files in `input_docs/` directory. Choose the appropriate script:

```bash
# For Markdown, TXT, RST files (pass-through + metadata)
uv run python 01_prepare_markdowndocs.py

# For Office, PDF, media files (requires conversion)
uv run python 01_prepare_officedocs.py

# Or use main.py
uv run python main.py prepare --source all
```

All supported file formats are converted to Markdown with metadata added.

#### PDF Processing Options

PDF files can be processed using two different libraries. Set the `PDF_PROCESSOR` environment variable to choose:

| Value | Library | Description |
| ----- | ------- | ----------- |
| `pymupdf4llm` (default) | PyMuPDF4LLM | LLM-optimized extraction, better table/structure preservation, automatic line break normalization |
| `markitdown` | Microsoft MarkItDown | Azure AI integration support (Document Intelligence OCR) |

```bash
# Use default (pymupdf4llm)
uv run python 01_prepare_officedocs.py

# Use markitdown (for Azure AI integration)
PDF_PROCESSOR=markitdown uv run python 01_prepare_officedocs.py
```

Or add to `.env` file:

```env
PDF_PROCESSOR=pymupdf4llm  # or "markitdown"
```

**When to use each option:**

- **pymupdf4llm** (recommended): Best for most PDFs, especially those with complex layouts, tables, or Korean text. Automatically removes unnecessary line breaks caused by PDF page layouts.
- **markitdown**: Use when Azure Document Intelligence OCR is needed for scanned PDFs or images.

#### 2. LLM Enrichment (Optional)

Enrich documents with AI-generated metadata:

```bash
uv run python 02_enrich_content.py
```

Requires environment variables:
- `ENRICHMENT_ENDPOINT`: Azure OpenAI endpoint
- `ENRICHMENT_API_KEY`: API key
- `ENRICHMENT_MODEL`: Model name (e.g., gpt-4.1)

If not set, this step is automatically skipped.

### 3. Structure-based Chunking

```bash
uv run python 03_chunk_content.py
```

Options:

- `--input-dir`: Input directory (default: `enriched_contents` or `prepared_contents`)
- `--output-dir`: Output directory (default: `chunked_data`)
- `--max-chunk-size`: Maximum chunk size (default: 1000)
- `--min-chunk-size`: Minimum chunk size (default: 50)

### 4. Application Example: sqlite-vec + MCP

```bash
# Build vector DB from Parquet
uv run python example_sqlitevec_mcp.py build

# Run MCP server (stdio mode)
uv run python example_sqlitevec_mcp.py serve

# Run MCP server (SSE mode)
uv run python example_sqlitevec_mcp.py serve --sse --port 8080

# Build and serve in one command
uv run python example_sqlitevec_mcp.py all
```

Options:

- `--input-dir`: Input directory (default: `chunked_data`)
- `--output-dir`: Output directory (default: `vector_db`)
- `--db-name`: DB filename (default: `vectors.db`)
- `--model`: Embedding model (default: `BAAI/bge-m3`)
- `--force`: Force rebuild if model mismatch
- `--export-parquet`: Export parquet for Milvus/Qdrant migration
- `--test-search "query"`: Perform test search after build
- `--sse`: Run in SSE mode
- `--host`: SSE server host (default: `127.0.0.1`)
- `--port`: SSE server port (default: `8080`)
- `--no-preload`: Don't preload model at startup

#### Vector DB Portability

Exported parquet files (`vectors_export.parquet`) can be directly imported to the following vector DBs:

| Vector DB | Import Method |
| --------- | ------------- |
| **Milvus** | Direct import using `pymilvus`'s `insert()` method |
| **Qdrant** | Upsert via REST API or Python client |
| **Pinecone** | Direct import using `upsert()` method |
| **Chroma** | Direct import using `add()` method |

Vector format: `float32[1024]` (PIXIE-Rune Dense vectors)

### 5. MCP Server Execution

Provides vector search via MCP protocol.

#### stdio Mode (Claude Desktop, Cursor, etc.)

```bash
uv run python example_sqlitevec_mcp.py serve
```

#### SSE Mode (Web clients)

```bash
uv run python example_sqlitevec_mcp.py serve --sse --port 8080
```

#### Available Tools

| Tool | Description |
| ---- | ----------- |
| `search` | Vector similarity search + reranking |
| `get_chunk` | Detailed lookup by chunk ID |
| `list_documents` | Document list lookup |
| `get_stats` | DB statistics lookup |

#### Claude Desktop Configuration Example

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reconsidered-rag-search": {
      "command": "uv",
      "args": ["run", "python", "example_sqlitevec_mcp.py", "serve"],
      "cwd": "D:/Projects/reconsidered_rag"
    }
  }
}
```

---

## Testing MCP Server in IDEs

This repository includes pre-configured MCP settings for VS Code and Cursor. Simply open the project folder to automatically connect to the MCP server.

### Supported IDEs

| IDE | Configuration File | Requirements |
| --- | ------------------ | ------------ |
| **VS Code + GitHub Copilot** | `.vscode/mcp.json` | GitHub Copilot Chat extension |
| **Cursor** | `.cursor/mcp.json` | Built-in MCP support |

### Quick Start

1. **Install dependencies**:

   ```bash
   uv sync
   ```

2. **Prepare sample data** (or add your own documents to `input_docs/`):

   ```bash
   uv run python 01_prepare_markdowndocs.py  # for .md, .txt, .rst
   uv run python 01_prepare_officedocs.py    # for .docx, .pdf, .pptx, etc.
   uv run python 02_enrich_content.py        # optional
   uv run python 03_chunk_content.py
   uv run python example_sqlitevec_mcp.py build
   ```

3. **Open the project folder** in VS Code or Cursor

4. **Start using MCP tools** in the chat:
   - The MCP server starts automatically when you open the folder
   - Available tools: `search`, `get_chunk`, `list_documents`, `get_stats`

### Configuration Files

**VS Code** (`.vscode/mcp.json`):

```json
{
  "servers": {
    "reconsidered-rag-search": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "example_sqlitevec_mcp.py", "serve"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "reconsidered-rag-search": {
      "command": "uv",
      "args": ["run", "python", "example_sqlitevec_mcp.py", "serve"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

### Troubleshooting

| Issue | Solution |
| ----- | -------- |
| MCP server not starting | Ensure `uv sync` was run and models are downloaded |
| Search returns errors | Restart the MCP server (VS Code: `Ctrl+Shift+P` → `MCP: Restart Server`) |
| No vector DB found | Run steps 2-4 to build the vector database |
| Model loading slow | First run downloads ~5GB of models; subsequent runs use cache |

---

## Output Schema

| Field | Type | Description |
| ----- | ---- | ----------- |
| `chunk_id` | string | Unique chunk ID |
| `content_hash` | string | Content hash (for incremental updates) |
| `chunk_text` | string | Chunk text |
| `chunk_type` | string | Type (header, paragraph, list, code, table) |
| `heading_level` | int32 | Heading level (0=normal, 1-6=H1-H6) |
| `heading_text` | string | Current heading text |
| `parent_heading` | string | Parent heading text |
| `section_path` | list[string] | Section hierarchy path array |
| `table_headers` | list[string] | Table column headers (if table) |
| `table_row_count` | int32 | Table data row count (if table) |
| `domain` | string | Domain (metadata) |
| `keywords` | string | Keywords JSON (metadata) |
| `version` | int32 | Version number |

---

## Directory Structure

```text
reconsidered_rag/
├── prepare_utils.py           # Common utilities for prepare scripts
├── 01_prepare_markdowndocs.py # Markdown/TXT/RST → Markdown
├── 01_prepare_officedocs.py   # Office/PDF/media → Markdown
├── 02_enrich_content.py       # LLM-based content enrichment
├── 03_chunk_content.py        # Structure-based chunking
├── example_sqlitevec_mcp.py   # Example: sqlite-vec + MCP server
├── input_docs/                # Input documents
├── prepared_contents/         # Step 1 output: Markdown
├── enriched_contents/         # Step 2 output: Enriched Markdown
├── chunked_data/              # Step 3 output: Parquet files
├── vector_db/                 # Example output: sqlite-vec DB
│   ├── vectors.db             # Local vector DB
│   └── vectors_export.parquet # Export file for migration
├── cache/
│   └── huggingface/           # Model cache directory
├── .env.example               # Environment variable template
├── pyproject.toml
└── README.md
```

---

## System Requirements

- Python 3.11+
- ~5GB disk space (for PIXIE-Rune + reranker models + ONNX cache)
- 8GB+ RAM recommended
- GPU (optional): Automatically utilized if CUDA-compatible GPU available

---

## Future Extension Possibilities

> [!NOTE]
> These extensions are intentionally not implemented yet.
They are listed to document design intent, not roadmap commitments.

The current system is designed with the following extensions in mind:

| Extension Direction | Description | Current Status |
| ------------------- | ----------- | -------------- |
| **Graph RAG** | Ontology-based entity/relation extraction → Node/edge parquet generation | Design completed |
| **Hybrid Search** | Combination of keyword + vector + graph search | Vector + reranking implemented |
| **Homonym Handling** | Context distinction via domain-specific ontology mapping | Metadata-based |
| **Multiple Vector DBs** | Migration to Milvus, Qdrant, Pinecone, etc. | Parquet export supported |
