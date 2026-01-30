"""
main.py
Fast path CLI for Reconsidered RAG

Core pipeline:
    input_docs/ → prepared_contents/ → enriched_contents/ → chunked_data/*.parquet

This is NOT a fast RAG DB builder.
This is a tool for people who want to own their data.
"""

import typer
import subprocess
import sys
from pathlib import Path
from typing import Optional

app = typer.Typer(
    help="""
Reconsidered RAG — RAG-ready document preparation.

Fast path: input_docs/ → Parquet (text only, no embedding lock-in)

For people who:
  • Want to own their data in portable formats
  • Don't want to be locked into a specific embedding model
  • Don't want to be locked into a specific vector DB
  • Value human-readable checkpoints over black-box pipelines
"""
)

BASE_DIR = Path(__file__).parent


def _run_script(script_name: str, args: Optional[list[str]] = None) -> int:
    """Run a Python script with uv run."""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        typer.secho(f"Script not found: {script_path}", fg=typer.colors.RED)
        return 1
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(cmd)
    return result.returncode


@app.command()
def run(
    source: str = typer.Option(
        "all",
        "--source", "-s",
        help="Source type: 'markdown', 'office', or 'all'"
    ),
    enrich: bool = typer.Option(
        False,
        "--enrich", "-e",
        help="Enable LLM enrichment (requires Azure OpenAI)"
    ),
    input_dir: Optional[str] = typer.Option(
        None,
        "--input-dir", "-i",
        help="Input directory (default: input_docs)"
    ),
):
    """
    Fast path: Documents → Parquet in one command.
    
    This runs the core pipeline:
    
      1. Prepare: Convert documents to Markdown with metadata
      2. Enrich: (optional) Expand bullet points with LLM
      3. Chunk: Split by structure, output to Parquet
    
    Output: chunked_data/*.parquet (text only, ready for any embedding model)
    """
    typer.echo("=" * 60)
    typer.echo("Reconsidered RAG — Fast Path")
    typer.echo("=" * 60)
    typer.echo()
    
    # Build common args
    prepare_args = []
    if input_dir:
        prepare_args.extend(["--input-dir", input_dir])
    
    # Step 1: Prepare
    typer.secho("Step 1: Preparing documents...", fg=typer.colors.CYAN)
    
    if source in ("markdown", "all"):
        typer.echo("  → Processing Markdown/TXT/RST files...")
        ret = _run_script("01_prepare_markdowndocs.py", prepare_args)
        if ret != 0:
            typer.secho("  ⚠️ Markdown preparation had issues", fg=typer.colors.YELLOW)
    
    if source in ("office", "all"):
        typer.echo("  → Processing Office/PDF/media files...")
        ret = _run_script("01_prepare_officedocs.py", prepare_args)
        if ret != 0:
            typer.secho("  ⚠️ Office preparation had issues", fg=typer.colors.YELLOW)
    
    typer.echo()
    
    # Step 2: Enrich (optional)
    if enrich:
        typer.secho("Step 2: Enriching content with LLM...", fg=typer.colors.CYAN)
        ret = _run_script("02_enrich_content.py")
        if ret != 0:
            typer.secho("  ⚠️ Enrichment skipped or failed", fg=typer.colors.YELLOW)
        typer.echo()
    else:
        typer.echo("Step 2: Enrichment skipped (use --enrich to enable)")
        typer.echo()
    
    # Step 3: Chunk
    typer.secho("Step 3: Chunking by structure...", fg=typer.colors.CYAN)
    ret = _run_script("03_chunk_content.py")
    if ret != 0:
        typer.secho("Chunking failed!", fg=typer.colors.RED)
        raise typer.Exit(1)
    
    typer.echo()
    typer.echo("=" * 60)
    typer.secho("Done! Your data is ready.", fg=typer.colors.GREEN)
    typer.echo("=" * 60)
    typer.echo()
    typer.echo("Output:")
    typer.echo("  • chunked_data/*.parquet  — Text chunks, ready for embedding")
    typer.echo()
    typer.echo("Next steps (YOUR choice):")
    typer.echo("  • Embed with OpenAI, Cohere, or local ONNX model")
    typer.echo("  • Import to Pinecone, Qdrant, Milvus, or any vector DB")
    typer.echo("  • Or try the included example: uv run python example_sqlitevec_mcp.py all")
    typer.echo()


@app.command()
def prepare(
    source: str = typer.Option(
        "all",
        "--source", "-s",
        help="Source type: 'markdown', 'office', or 'all'"
    ),
    input_dir: Optional[str] = typer.Option(
        None,
        "--input-dir", "-i",
        help="Input directory"
    ),
):
    """Prepare documents: Convert to Markdown with metadata."""
    args = []
    if input_dir:
        args.extend(["--input-dir", input_dir])
    
    if source in ("markdown", "all"):
        typer.echo("Processing Markdown/TXT/RST...")
        _run_script("01_prepare_markdowndocs.py", args)
    
    if source in ("office", "all"):
        typer.echo("Processing Office/PDF/media...")
        _run_script("01_prepare_officedocs.py", args)


@app.command()
def enrich():
    """Enrich content: Expand bullet points with LLM (requires Azure OpenAI)."""
    raise typer.Exit(_run_script("02_enrich_content.py"))


@app.command()
def chunk(
    input_dir: Optional[str] = typer.Option(None, "--input-dir"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
):
    """Chunk content: Split by structure, output to Parquet."""
    args = []
    if input_dir:
        args.extend(["--input-dir", input_dir])
    if output_dir:
        args.extend(["--output-dir", output_dir])
    raise typer.Exit(_run_script("03_chunk_content.py", args))


@app.command()
def cleanup(
    include_cache: bool = typer.Option(
        False, "--include-cache",
        help="Also clean cache directory"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Skip confirmation prompt"
    ),
):
    """Clean up output directories."""
    args = []
    if include_cache:
        args.append("--include-cache")
    if force:
        args.append("--force")
    raise typer.Exit(_run_script("99_cleanup.py", args))


def main():
    app()


if __name__ == "__main__":
    main()
