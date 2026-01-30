"""
01_prepare_officedocs.py
Prepare Office documents, PDFs, and media files

This script handles binary documents that require conversion:
- Office: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- PDF files
- Images (EXIF/OCR)
- Audio (speech-to-text)
- Video (subtitle extraction)
- Code files, Jupyter notebooks, archives

Uses:
- pymupdf4llm for PDF conversion (LLM-optimized, table/structure preservation)
- Microsoft MarkItDown for Office and media files
- Optional Azure AI services for enhanced OCR and image understanding

Usage:
    uv run python 01_prepare_officedocs.py
    uv run python 01_prepare_officedocs.py --input-dir ./my_docs
    uv run python 01_prepare_officedocs.py --pdf-processor markitdown
"""

import os
import re
from pathlib import Path
from typing import Optional, Any

import pymupdf4llm
from markitdown import MarkItDown, UnsupportedFormatException

from prepare_utils import (
    BASE_DIR,
    OUTPUT_DIR,
    CIRCLED_NUMBERS,
    LEFT_BLACK_LENTICULAR_BRACKET,
    get_azure_document_intelligence_client,
    get_azure_openai_client,
    create_front_matter,
    strip_existing_front_matter,
    write_with_front_matter,
    print_processing_result,
)


# Supported extensions for this script
SUPPORTED_EXTENSIONS = {
    # Office documents
    ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    # PDF
    ".pdf",
    # Web/structured
    ".html", ".htm", ".xml", ".json", ".csv",
    # Images (EXIF/OCR)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff",
    # Audio (speech recognition)
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    # Video (subtitle extraction)
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    # Code/other
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs",
    ".ipynb",  # Jupyter Notebook
    ".zip",    # Archive
}

# Default input directory
INPUT_DIR = BASE_DIR / "input_docs"

# PDF processor setting
PDF_PROCESSOR = os.getenv("PDF_PROCESSOR", "pymupdf4llm")

# MarkItDown singleton
_markitdown_instance: Optional[MarkItDown] = None


def get_markitdown() -> MarkItDown:
    """
    Get MarkItDown singleton instance
    
    Automatically integrates Azure services if configured:
    - Document Intelligence: Enhanced OCR for scanned PDFs and images
    - OpenAI (GPT-4o): Image content understanding
    """
    global _markitdown_instance
    if _markitdown_instance is None:
        doc_client = get_azure_document_intelligence_client()
        llm_client = get_azure_openai_client()
        llm_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        services = []
        if doc_client:
            services.append("Document Intelligence")
        if llm_client and llm_model:
            services.append(f"OpenAI ({llm_model})")
        
        if services:
            print(f"🔗 Azure 서비스 연동: {', '.join(services)}")
        else:
            print("ℹ️ Azure 서비스 미연동 (기본 markitdown 사용)")
        
        _markitdown_instance = MarkItDown(
            document_intelligence_client=doc_client,
            llm_client=llm_client,
            llm_model=llm_model,
        )
    
    return _markitdown_instance


def is_supported_file(file_path: Path) -> bool:
    """Check if file is supported by this script"""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def normalize_line_breaks(text: str) -> str:
    """
    Clean up unnecessary line breaks from PDF extraction.
    
    PDFs wrap text based on page layout, causing mid-sentence line breaks.
    This function removes those while preserving paragraph boundaries.
    """
    lines = text.split('\n')
    result_lines = []
    current_paragraph = []
    pending_empty_lines = 0

    left_bracket = LEFT_BLACK_LENTICULAR_BRACKET
    new_block_pattern = re.compile(
        r'^('
        rf'\uc81c\s*\d+\s*(\uc870|\uad00|\uc7a5|\uc808|\ud3b8)\s*{left_bracket}|'
        r'\uc81c\s*\d+\s*(\uc870|\uad00|\uc7a5|\uc808|\ud3b8)\s*$|'
        r'\(\ubcc4\ud45c\s*\d+\)|'
        r'\(\ubcc4\s*\ud45c\s*\d*\)'
        r')'
    )

    circled_pattern = ''.join(CIRCLED_NUMBERS)
    item_pattern = re.compile(rf'^([{circled_pattern}]|\d+\.)\s*')

    def is_markdown_structure(stripped: str) -> bool:
        if not stripped:
            return False
        return (stripped.startswith('#') or
                stripped.startswith('|') or
                stripped.startswith('```') or
                stripped == '---' or stripped == '-----' or
                stripped == '===')

    def is_new_section(stripped: str) -> bool:
        if not stripped:
            return False
        if is_markdown_structure(stripped):
            return True
        if new_block_pattern.match(stripped):
            return True
        if item_pattern.match(stripped):
            return True
        return False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            pending_empty_lines += 1
            continue

        if pending_empty_lines > 0:
            if is_new_section(stripped):
                if current_paragraph:
                    result_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                if pending_empty_lines >= 2:
                    result_lines.append('')
            pending_empty_lines = 0

        if is_markdown_structure(stripped):
            if current_paragraph:
                result_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            result_lines.append(stripped)
            continue

        if new_block_pattern.match(stripped):
            if current_paragraph:
                result_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            result_lines.append(stripped)
            continue

        if item_pattern.match(stripped):
            if current_paragraph:
                result_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            current_paragraph.append(stripped)
            continue

        current_paragraph.append(stripped)

    if current_paragraph:
        result_lines.append(' '.join(current_paragraph))

    return '\n'.join(result_lines)


def convert_pdf_pymupdf(file_path: Path) -> str:
    """Convert PDF using pymupdf4llm"""
    result = pymupdf4llm.to_markdown(str(file_path))
    
    if isinstance(result, list):
        markdown_content = "\n\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in result
        )
    else:
        markdown_content = result
    
    return normalize_line_breaks(markdown_content)


def convert_pdf_markitdown(file_path: Path) -> str:
    """Convert PDF using markitdown (with optional Azure OCR)"""
    md = get_markitdown()
    result = md.convert(str(file_path))
    
    if result.text_content:
        return result.text_content
    
    raise ValueError(f"PDF 변환 결과가 비어있습니다: {file_path}")


def convert_to_markdown(file_path: Path, pdf_processor: Optional[str] = None) -> tuple[str, str]:
    """
    Convert file to markdown
    
    Args:
        file_path: Input file path
        pdf_processor: PDF processor ("pymupdf4llm" or "markitdown")
    
    Returns:
        (markdown_content, source_format) tuple
    """
    suffix = file_path.suffix.lower()
    
    # PDF files
    if suffix == ".pdf":
        processor = pdf_processor or PDF_PROCESSOR
        if processor == "markitdown":
            content = convert_pdf_markitdown(file_path)
        else:
            content = convert_pdf_pymupdf(file_path)
        return content, "pdf"
    
    # Other files via MarkItDown
    md = get_markitdown()
    result = md.convert(str(file_path))
    
    if result.text_content:
        return result.text_content, suffix.lstrip(".")
    
    raise ValueError(f"변환 결과가 비어있습니다: {file_path}")


def prepare_document(
    input_path: Path,
    output_path: Optional[Path] = None,
    pdf_processor: Optional[str] = None,
) -> Path:
    """
    Prepare document with YAML front matter
    
    Args:
        input_path: Input file path
        output_path: Output path (default: OUTPUT_DIR)
        pdf_processor: PDF processor override
    
    Returns:
        Created file path
    """
    # Convert to markdown
    content, source_format = convert_to_markdown(input_path, pdf_processor)
    
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
    pdf_processor: Optional[str] = None,
) -> list[Path]:
    """
    Process all supported documents
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        pdf_processor: PDF processor override
    
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
    
    print(f"\n📄 Office/PDF/미디어 문서 처리: {len(all_files)}개")
    print(f"   형식별: {', '.join(f'{ext}({cnt})' for ext, cnt in sorted(format_counts.items()))}")
    print(f"   PDF 처리기: {pdf_processor or PDF_PROCESSOR}")
    print("=" * 50)
    
    results = []
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] {file_path.name}")
        
        try:
            output_path = output_dir / file_path.with_suffix('.md').name
            result = prepare_document(file_path, output_path, pdf_processor)
            
            # Read back metadata for display
            content = result.read_text(encoding="utf-8")
            if content.startswith("---"):
                import yaml
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    meta = yaml.safe_load(parts[1])
                    print_processing_result(result, meta)
            
            results.append(result)
            
        except UnsupportedFormatException as e:
            print(f"   ⚠️ 지원하지 않는 형식: {e}")
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
        description="Office, PDF, 미디어 문서를 Markdown으로 변환하고 메타데이터를 추가합니다."
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
    parser.add_argument(
        "--pdf-processor",
        type=str,
        choices=["pymupdf4llm", "markitdown"],
        default=None,
        help=f"PDF 처리기 (기본값: 환경변수 PDF_PROCESSOR 또는 pymupdf4llm)",
    )
    
    args = parser.parse_args()
    
    results = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pdf_processor=args.pdf_processor,
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[중단됨]")
        exit(130)
