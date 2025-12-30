#!/bin/bash
# Entrypoint script for aipack container
# Executes model download, data processing, and MCP server startup sequentially

set -e

echo "🚀 Starting aipack container..."

# 1. Model download (if not cached)
echo "📥 Checking model cache..."
if [ ! -d "/app/cache/huggingface/models--BAAI--bge-m3" ] || [ ! -d "/app/cache/huggingface/models--BAAI--bge-reranker-large" ]; then
    echo "🔄 Downloading models..."
    python 01_download_model.py
else
    echo "✅ Models already cached"
fi

# 2. Data processing (if input_docs exists and vector_db doesn't)
if [ -d "/app/input_docs" ] && [ ! -f "/app/vector_db/vectors.db" ]; then
    echo "🔄 Processing documents..."
    # Run sequential processing
    echo "   • Preparing content..."
    python 02_prepare_content.py
    echo "   • Semantic chunking..."
    python 03_semantic_chunking.py
    echo "   • Building vector DB..."
    python 04_build_vector_db.py
    echo "✅ Data processing complete"
else
    echo "ℹ️  Skipping data processing (no input docs or DB exists)"
fi

# 3. Start MCP server
echo "🌐 Starting MCP server..."
if [ "$1" = "sse" ]; then
    echo "   Mode: SSE on port ${PORT:-8080}"
    exec python 05_mcp_server.py --sse --host 0.0.0.0 --port ${PORT:-8080}
else
    echo "   Mode: stdio"
    exec python 05_mcp_server.py
fi