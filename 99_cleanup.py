"""
99_cleanup.py
Module for cleaning up output directories (except input_docs)

Features:
- Delete all files in prepared_contents, chunked_data, vector_db directories
- Optionally delete cache directory contents
- Preserve directory structure (only delete files)
"""

import shutil
from pathlib import Path

import typer

# 디렉터리 설정
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_docs"

# 정리 대상 디렉터리
OUTPUT_DIRS = [
    BASE_DIR / "prepared_contents",
    BASE_DIR / "chunked_data",
    BASE_DIR / "enriched_contents",
    BASE_DIR / "vector_db",
    BASE_DIR / "test_output",
]

CACHE_DIR = BASE_DIR / "cache"


def cleanup_directory(dir_path: Path, verbose: bool = True) -> int:
    """
    Delete all files and subdirectories in the specified directory.
    The directory itself is preserved.

    Args:
        dir_path: Path to the directory to clean
        verbose: Whether to print progress messages

    Returns:
        Number of items deleted
    """
    if not dir_path.exists():
        if verbose:
            typer.echo(f"  [SKIP] Directory does not exist: {dir_path}")
        return 0

    deleted_count = 0
    for item in dir_path.iterdir():
        try:
            if item.is_file():
                item.unlink()
                deleted_count += 1
                if verbose:
                    typer.echo(f"  [DEL] File: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                deleted_count += 1
                if verbose:
                    typer.echo(f"  [DEL] Directory: {item.name}/")
        except PermissionError as e:
            typer.secho(f"  [ERR] Permission denied: {item} - {e}", fg=typer.colors.RED)
        except OSError as e:
            typer.secho(f"  [ERR] Failed to delete: {item} - {e}", fg=typer.colors.RED)

    return deleted_count


def main(
    include_cache: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> int:
    """
    Clean up output directories.

    Args:
        include_cache: Whether to also delete cache directory contents
        force: Skip confirmation prompt
        verbose: Whether to print progress messages

    Returns:
        0 on success, 1 on error
    """
    typer.echo("=" * 60)
    typer.echo("Output Directory Cleanup")
    typer.echo("=" * 60)

    # 정리 대상 목록 표시
    target_dirs = OUTPUT_DIRS.copy()
    if include_cache:
        target_dirs.append(CACHE_DIR)

    typer.echo("\nTarget directories:")
    for d in target_dirs:
        status = "EXISTS" if d.exists() else "NOT FOUND"
        typer.echo(f"  - {d.relative_to(BASE_DIR)} [{status}]")

    # 확인 프롬프트
    if not force:
        typer.echo("")
        confirm = typer.confirm("Are you sure you want to delete all files in these directories?")
        if not confirm:
            typer.echo("Cancelled.")
            return 0

    typer.echo("\nCleaning up...")
    total_deleted = 0

    for dir_path in target_dirs:
        typer.echo(f"\n[{dir_path.relative_to(BASE_DIR)}]")
        deleted = cleanup_directory(dir_path, verbose=verbose)
        total_deleted += deleted
        if deleted > 0:
            typer.secho(f"  → Deleted {deleted} item(s)", fg=typer.colors.GREEN)
        else:
            typer.echo("  → Nothing to delete")

    typer.echo("\n" + "=" * 60)
    typer.secho(f"Cleanup complete! Total deleted: {total_deleted} item(s)", fg=typer.colors.GREEN)
    typer.echo("=" * 60)

    return 0


if __name__ == "__main__":
    typer.run(main)
