import typer
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional
import sys

app = typer.Typer(help="aipack gateway CLI")


def _load_module_from_file(filename: str) -> ModuleType:
    """Load a Python module from a filename relative to the project root."""
    base = Path(__file__).parent
    path = base / filename
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def _call_main(module: ModuleType, args: Optional[list] = None) -> int:
    """Call module.main() while temporarily replacing sys.argv to avoid argument conflicts."""
    old_argv = sys.argv
    try:
        sys.argv = [module.__file__]
        if args:
            sys.argv.extend(args)
        if hasattr(module, "main"):
            return module.main()
        raise RuntimeError("Module has no main()")
    finally:
        sys.argv = old_argv


@app.command()
def download_model():
    """BGE-M3 임베딩 모델 다운로드"""
    mod = _load_module_from_file("01_download_model.py")
    # prefer named function if available
    if hasattr(mod, "download_model"):
        result = mod.download_model()
        typer.echo(f"Model download result: {result}")
    elif hasattr(mod, "main"):
        res = mod.main()
        typer.echo(f"Exited with: {res}")
    else:
        raise RuntimeError("No entrypoint found in 01_download_model.py")


@app.command()
def prepare_content():
    """Extract metadata from input documents and add YAML front matter"""
    mod = _load_module_from_file("02_prepare_content.py")
    if hasattr(mod, "main"):
        raise SystemExit(_call_main(mod))
    raise RuntimeError("No main() in 02_prepare_content.py")


@app.command()
def semantic_chunking():
    """Perform semantic chunking on markdown files from prepared_contents and save as parquet"""
    mod = _load_module_from_file("03_semantic_chunking.py")
    if hasattr(mod, "main"):
        raise SystemExit(_call_main(mod))
    raise RuntimeError("No main() in 03_semantic_chunking.py")


@app.command()
def build_vector_db():
    """chunked_data의 parquet 파일을 읽어 벡터 DB 구축"""
    mod = _load_module_from_file("04_build_vector_db.py")
    if hasattr(mod, "main"):
        raise SystemExit(_call_main(mod))
    raise RuntimeError("No main() in 04_build_vector_db.py")


@app.command()
def mcp_server(sse: bool = typer.Option(False, help="SSE 모드"), port: int = typer.Option(8080, help="서버 포트")):
    """MCP 서버 실행"""
    mod = _load_module_from_file("05_mcp_server.py")
    if hasattr(mod, "main"):
        # pass CLI args via argv so internal arg parsing (if any) can work
        args = ["--sse"] if sse else ["--port", str(port)]
        try:
            raise SystemExit(_call_main(mod, args=args))
        except SystemExit as exc:
            if exc.code != 0:
                typer.secho(
                    "MCP server exited with non-zero status. If you see a port bind error, try a different --port or stop the process using that port.",
                    fg=typer.colors.RED,
                )
            raise
    raise RuntimeError("No main() in 05_mcp_server.py")


def main():
    app()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
