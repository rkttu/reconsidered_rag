"""
01_prepare_markdowndocs.py
Prepare Markdown, Text, and RST documents

This script handles text-based documents that are already human-readable:
- Markdown (.md, .markdown)
- Plain text (.txt)
- reStructuredText (.rst)

These files require minimal processing - mainly adding YAML front matter
with extracted metadata.

Usage:
    uv run python 01_prepare_markdowndocs.py
    uv run python 01_prepare_markdowndocs.py --input-dir ./my_docs
"""

from pathlib import Path
from typing import Optional

from prepare_utils import (
    BASE_DIR,
    OUTPUT_DIR,
    create_front_matter,
    strip_existing_front_matter,
    write_with_front_matter,
    print_processing_result,
)


# Supported extensions for this script
SUPPORTED_EXTENSIONS = {
    ".md", ".markdown",  # Markdown
    ".txt",              # Plain text
    ".rst",              # reStructuredText
}

# Default input directory for markdown docs
INPUT_DIR = BASE_DIR / "input_docs"


def is_supported_file(file_path: Path) -> bool:
    """Check if file is supported by this script"""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def convert_to_markdown(file_path: Path) -> tuple[str, str]:
    """
    Convert text file to markdown
    
    For markdown and text files, this is essentially a pass-through.
    
    Args:
        file_path: Input file path
    
    Returns:
        (content, source_format) tuple
    """
    suffix = file_path.suffix.lower()
    content = file_path.read_text(encoding="utf-8")
    
    if suffix in {".md", ".markdown"}:
        return content, "markdown"
    elif suffix == ".txt":
        return content, "plaintext"
    elif suffix == ".rst":
        # RST is kept as-is for now (could add conversion later)
        return content, "rst"
    else:
        return content, suffix.lstrip(".")


def prepare_document(
    input_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Prepare document with YAML front matter
    
    Args:
        input_path: Input file path
        output_path: Output path (default: OUTPUT_DIR)
    
    Returns:
        Created file path
    """
    # Convert to markdown
    content, source_format = convert_to_markdown(input_path)
    
    # Strip existing front matter
    original_content = strip_existing_front_matter(content)
    
    # Create metadata
    metadata = create_front_matter(
        content=original_content,
        source_file=input_path.name,
        source_format=source_format,
    )
    
    # Determine output path
    if output_path is None:
        output_path = OUTPUT_DIR / input_path.with_suffix('.md').name
    
    # Write with front matter
    return write_with_front_matter(output_path, original_content, metadata)


def process_all_documents(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """
    Process all supported documents
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
    
    Returns:
        List of created file paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"⚠️ 입력 디렉터리가 없습니다: {input_dir}")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect supported files
    all_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and is_supported_file(f)
    ]
    
    if not all_files:
        print(f"⚠️ 처리할 파일이 없습니다: {input_dir}")
        print(f"   지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return []
    
    # Format statistics
    format_counts: dict[str, int] = {}
    for f in all_files:
        ext = f.suffix.lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
    
    print(f"\n📝 Markdown/Text 문서 처리: {len(all_files)}개")
    print(f"   형식별: {', '.join(f'{ext}({cnt})' for ext, cnt in sorted(format_counts.items()))}")
    print("=" * 50)
    
    results = []
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] {file_path.name}")
        
        try:
            output_path = output_dir / file_path.with_suffix('.md').name
            result = prepare_document(file_path, output_path)
            
            # Read back metadata for display
            content = result.read_text(encoding="utf-8")
            if content.startswith("---"):
                import yaml
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    meta = yaml.safe_load(parts[1])
                    print_processing_result(result, meta)
            
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
    
    print("\n" + "=" * 50)
    print(f"✅ 완료: {len(results)}/{len(all_files)} 문서 처리됨")
    print(f"📁 출력 위치: {output_dir}")
    
    return results


def main() -> int:
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Markdown, Text, RST 문서에 YAML front matter를 추가합니다."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"입력 디렉터리 (기본값: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"출력 디렉터리 (기본값: {OUTPUT_DIR})",
    )
    
    args = parser.parse_args()
    
    results = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[중단됨]")
        exit(130)
