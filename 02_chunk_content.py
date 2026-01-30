"""
02_chunk_content.py
Module that splits markdown files from prepared_contents using structure-based chunking
and saves as parquet files

Features:
- Markdown structure parsing (preserves heading hierarchy)
- Structure-based chunking (headings, paragraphs, lists, tables, code blocks)
- No embedding model required - fast and deterministic
- zstd compression and incremental update support
"""

import re
import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import mistune


# Directory settings
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "prepared_contents"
OUTPUT_DIR = BASE_DIR / "chunked_data"

# Chunking settings
MAX_CHUNK_SIZE = 2000  # Maximum chunk size (characters)
MIN_CHUNK_SIZE = 100   # Minimum chunk size (characters)
OVERLAP_SIZE = 100     # Overlap size for split paragraphs (characters)


@dataclass
class MarkdownSection:
    """Markdown section information"""
    text: str
    heading_level: int = 0  # 0 = regular text, 1-6 = heading level
    heading_text: str = ""
    section_path: list[str] = field(default_factory=list)
    element_type: str = "paragraph"  # header, paragraph, list, code, blockquote, table
    line_start: int = 0
    line_end: int = 0
    # Table-specific metadata
    table_headers: list[str] = field(default_factory=list)
    table_row_count: int = 0


@dataclass
class StructureChunk:
    """Structure-based chunk information"""
    text: str
    heading_level: int = 0
    heading_text: str = ""
    parent_heading: str = ""
    section_path: list[str] = field(default_factory=list)
    chunk_type: str = "paragraph"
    start_line: int = 0
    end_line: int = 0
    # Table-specific metadata
    table_headers: list[str] = field(default_factory=list)
    table_row_count: int = 0


class MarkdownParser:
    """Mistune-based markdown parser"""
    
    def __init__(self):
        self.sections: list[MarkdownSection] = []
        self.current_headings: dict[int, str] = {}
    
    def parse(self, markdown_text: str) -> list[MarkdownSection]:
        """
        Parse markdown and return section list
        
        Args:
            markdown_text: Markdown text
            
        Returns:
            List of MarkdownSection
        """
        self.sections = []
        self.current_headings = {}
        
        # Parse with mistune (enable table plugin)
        md = mistune.create_markdown(renderer=None, plugins=['table'])
        tokens = md(markdown_text)
        
        if tokens is None:
            tokens = []
        
        for token in tokens:
            if isinstance(token, dict):
                self._process_token(token)
        
        # Fallback to line-based parsing if token parsing fails
        if not self.sections:
            self._fallback_parse(markdown_text)
        
        return self.sections
    
    def _process_token(self, token: dict) -> None:
        """Process a single token"""
        token_type = token.get('type', '')
        
        if token_type == 'heading':
            level = token.get('attrs', {}).get('level', 1)
            children = token.get('children', [])
            text = self._extract_text_from_children(children)
            
            # Update current heading
            self.current_headings[level] = text
            # Clear lower level headings
            for lv in range(level + 1, 7):
                self.current_headings.pop(lv, None)
            
            section_path = self._build_section_path()
            
            self.sections.append(MarkdownSection(
                text=f"{'#' * level} {text}",
                heading_level=level,
                heading_text=text,
                section_path=section_path,
                element_type="header",
            ))
        
        elif token_type == 'paragraph':
            children = token.get('children', [])
            text = self._extract_text_from_children(children)
            if text.strip():
                self.sections.append(MarkdownSection(
                    text=text,
                    heading_level=0,
                    heading_text=self._get_current_heading(),
                    section_path=self._build_section_path(),
                    element_type="paragraph",
                ))
        
        elif token_type == 'list':
            items = token.get('children', [])
            list_text = self._extract_list_text(items)
            if list_text.strip():
                self.sections.append(MarkdownSection(
                    text=list_text,
                    heading_level=0,
                    heading_text=self._get_current_heading(),
                    section_path=self._build_section_path(),
                    element_type="list",
                ))
        
        elif token_type == 'code_block':
            raw = token.get('raw', '')
            if raw.strip():
                self.sections.append(MarkdownSection(
                    text=raw,
                    heading_level=0,
                    heading_text=self._get_current_heading(),
                    section_path=self._build_section_path(),
                    element_type="code",
                ))
        
        elif token_type == 'block_quote':
            children = token.get('children', [])
            text = self._extract_text_from_children(children)
            if text.strip():
                self.sections.append(MarkdownSection(
                    text=text,
                    heading_level=0,
                    heading_text=self._get_current_heading(),
                    section_path=self._build_section_path(),
                    element_type="blockquote",
                ))
        
        elif token_type == 'table':
            table_text, headers, row_count = self._extract_table(token)
            if table_text.strip():
                self.sections.append(MarkdownSection(
                    text=table_text,
                    heading_level=0,
                    heading_text=self._get_current_heading(),
                    section_path=self._build_section_path(),
                    element_type="table",
                    table_headers=headers,
                    table_row_count=row_count,
                ))
    
    def _extract_text_from_children(self, children: list) -> str:
        """Extract text from children tokens"""
        texts = []
        for child in children:
            if isinstance(child, dict):
                if child.get('type') == 'text':
                    texts.append(child.get('raw', ''))
                elif 'children' in child:
                    texts.append(self._extract_text_from_children(child['children']))
                elif 'raw' in child:
                    texts.append(child.get('raw', ''))
        return ''.join(texts)
    
    def _extract_list_text(self, items: list) -> str:
        """Extract text from list items"""
        texts = []
        for item in items:
            if isinstance(item, dict):
                children = item.get('children', [])
                item_text = self._extract_text_from_children(children)
                texts.append(f"- {item_text}")
        return '\n'.join(texts)
    
    def _extract_table(self, token: dict) -> tuple[str, list[str], int]:
        """
        Extract text, headers, and row count from table token
        
        Returns:
            (table_markdown, headers, row_count)
        """
        headers: list[str] = []
        rows: list[list[str]] = []
        
        children = token.get('children', [])
        for child in children:
            if not isinstance(child, dict):
                continue
            
            child_type = child.get('type', '')
            
            if child_type == 'table_head':
                head_cells = child.get('children', [])
                for cell in head_cells:
                    if isinstance(cell, dict) and cell.get('type') == 'table_cell':
                        cell_text = self._extract_text_from_children(
                            cell.get('children', [])
                        )
                        headers.append(cell_text.strip())
            
            elif child_type == 'table_body':
                body_rows = child.get('children', [])
                for row in body_rows:
                    if isinstance(row, dict) and row.get('type') == 'table_row':
                        cells = row.get('children', [])
                        row_data = []
                        for cell in cells:
                            if isinstance(cell, dict) and cell.get('type') == 'table_cell':
                                cell_text = self._extract_text_from_children(
                                    cell.get('children', [])
                                )
                                row_data.append(cell_text.strip())
                        if row_data:
                            rows.append(row_data)
        
        # Reconstruct markdown table
        md_lines = []
        if headers:
            md_lines.append('| ' + ' | '.join(headers) + ' |')
            md_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        for row in rows:
            padded_row = row + [''] * (len(headers) - len(row)) if headers else row
            md_lines.append('| ' + ' | '.join(padded_row) + ' |')
        
        table_text = '\n'.join(md_lines)
        return table_text, headers, len(rows)
    
    def _build_section_path(self) -> list[str]:
        """Build current section path as array"""
        parts = []
        for level in sorted(self.current_headings.keys()):
            parts.append(self.current_headings[level])
        return parts
    
    def _get_current_heading(self) -> str:
        """Get the deepest current heading"""
        if self.current_headings:
            max_level = max(self.current_headings.keys())
            return self.current_headings[max_level]
        return ""
    
    def _fallback_parse(self, markdown_text: str) -> None:
        """Line-based fallback parsing"""
        lines = markdown_text.split('\n')
        current_headings: dict[int, str] = {}
        current_paragraph: list[str] = []
        
        def flush_paragraph():
            if current_paragraph:
                text = '\n'.join(current_paragraph).strip()
                if text:
                    section_path = [
                        h for _, h in sorted(current_headings.items())
                    ]
                    current_heading = current_headings.get(
                        max(current_headings.keys()) if current_headings else 0, ""
                    )
                    
                    is_list = all(
                        line.strip().startswith(('-', '*', '1.', '2.', '3.'))
                        for line in current_paragraph if line.strip()
                    )
                    
                    self.sections.append(MarkdownSection(
                        text=text,
                        heading_level=0,
                        heading_text=current_heading,
                        section_path=section_path,
                        element_type="list" if is_list else "paragraph",
                    ))
                current_paragraph.clear()
        
        def is_table_line(line: str) -> bool:
            s = line.strip()
            return s.startswith('|') and s.endswith('|')
        
        def is_separator_line(line: str) -> bool:
            s = line.strip()
            if not (s.startswith('|') and s.endswith('|')):
                return False
            return bool(re.match(r'^\|[\s\-:|]+\|$', s))
        
        def flush_table(table_lines: list[str], start_idx: int):
            if not table_lines:
                return
            
            table_text = '\n'.join(table_lines)
            headers: list[str] = []
            row_count = 0
            
            if table_lines:
                first_line = table_lines[0].strip()
                if first_line.startswith('|') and first_line.endswith('|'):
                    cells = [c.strip() for c in first_line[1:-1].split('|')]
                    headers = [c for c in cells if c]
            
            for tl in table_lines[2:]:
                if is_table_line(tl) and not is_separator_line(tl):
                    row_count += 1
            
            section_path = [
                h for _, h in sorted(current_headings.items())
            ]
            current_heading = current_headings.get(
                max(current_headings.keys()) if current_headings else 0, ""
            )
            
            self.sections.append(MarkdownSection(
                text=table_text,
                heading_level=0,
                heading_text=current_heading,
                section_path=section_path,
                element_type="table",
                line_start=start_idx,
                line_end=start_idx + len(table_lines) - 1,
                table_headers=headers,
                table_row_count=row_count,
            ))
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if heading_match:
                flush_paragraph()
                
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                current_headings[level] = heading_text
                for lv in list(current_headings.keys()):
                    if lv > level:
                        del current_headings[lv]
                
                section_path = [
                    h for _, h in sorted(current_headings.items())
                ]
                
                self.sections.append(MarkdownSection(
                    text=stripped,
                    heading_level=level,
                    heading_text=heading_text,
                    section_path=section_path,
                    element_type="header",
                    line_start=i,
                    line_end=i,
                ))
                i += 1
            
            elif is_table_line(stripped):
                flush_paragraph()
                
                table_lines = [line]
                table_start = i
                i += 1
                
                while i < len(lines) and is_table_line(lines[i].strip()):
                    table_lines.append(lines[i])
                    i += 1
                
                if len(table_lines) >= 2:
                    flush_table(table_lines, table_start)
                else:
                    current_paragraph.extend(table_lines)
            
            elif stripped == '':
                flush_paragraph()
                i += 1
            
            else:
                current_paragraph.append(line)
                i += 1
        
        flush_paragraph()


class StructureChunker:
    """Structure-based chunking (no embedding required)"""
    
    def __init__(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        min_chunk_size: int = MIN_CHUNK_SIZE,
        overlap_size: int = OVERLAP_SIZE,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.parser = MarkdownParser()
    
    def chunk_document(self, content: str, metadata: dict) -> list[StructureChunk]:
        """
        Split document into structure-based chunks
        
        Strategy:
        1. Respect heading boundaries (each heading starts a new chunk)
        2. Keep tables, code blocks, and lists intact if possible
        3. Split large paragraphs at sentence boundaries with overlap
        
        Args:
            content: Markdown document content
            metadata: Document metadata
            
        Returns:
            List of StructureChunk
        """
        # 1. Parse markdown
        sections = self.parser.parse(content)
        
        if not sections:
            return [StructureChunk(
                text=content,
                chunk_type="document",
            )]
        
        # 2. Group sections by heading
        chunks: list[StructureChunk] = []
        current_sections: list[MarkdownSection] = []
        current_size = 0
        
        for section in sections:
            section_size = len(section.text)
            
            # Headings always start a new chunk
            if section.heading_level > 0:
                # Save previous chunk
                if current_sections:
                    chunks.extend(self._finalize_chunk(current_sections))
                current_sections = [section]
                current_size = section_size
                continue
            
            # Tables, code blocks, and lists: keep intact if possible
            if section.element_type in ("table", "code", "list"):
                if section_size > self.max_chunk_size:
                    # Too large - save as separate chunk anyway
                    if current_sections:
                        chunks.extend(self._finalize_chunk(current_sections))
                    chunks.append(self._section_to_chunk(section))
                    current_sections = []
                    current_size = 0
                elif current_size + section_size > self.max_chunk_size:
                    # Would exceed max - save current and start new
                    if current_sections:
                        chunks.extend(self._finalize_chunk(current_sections))
                    current_sections = [section]
                    current_size = section_size
                else:
                    current_sections.append(section)
                    current_size += section_size
                continue
            
            # Regular paragraphs
            if current_size + section_size > self.max_chunk_size:
                # Would exceed max
                if current_sections:
                    chunks.extend(self._finalize_chunk(current_sections))
                
                # Check if this section itself is too large
                if section_size > self.max_chunk_size:
                    # Split large paragraph
                    chunks.extend(self._split_large_section(section))
                    current_sections = []
                    current_size = 0
                else:
                    current_sections = [section]
                    current_size = section_size
            else:
                current_sections.append(section)
                current_size += section_size
        
        # Save remaining sections
        if current_sections:
            chunks.extend(self._finalize_chunk(current_sections))
        
        return chunks
    
    def _finalize_chunk(self, sections: list[MarkdownSection]) -> list[StructureChunk]:
        """Convert sections to chunks, splitting if too large"""
        if not sections:
            return []
        
        total_size = sum(len(s.text) for s in sections)
        
        if total_size <= self.max_chunk_size:
            return [self._merge_sections(sections)]
        
        # Split by sections
        result = []
        current = []
        current_size = 0
        
        for section in sections:
            section_size = len(section.text)
            
            if current_size + section_size > self.max_chunk_size and current:
                result.append(self._merge_sections(current))
                current = [section]
                current_size = section_size
            else:
                current.append(section)
                current_size += section_size
        
        if current:
            result.append(self._merge_sections(current))
        
        return result
    
    def _split_large_section(self, section: MarkdownSection) -> list[StructureChunk]:
        """Split a large section at sentence boundaries with overlap"""
        text = section.text
        
        # Split by sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_text = ""
        
        for sentence in sentences:
            if len(current_text) + len(sentence) > self.max_chunk_size:
                if current_text:
                    chunks.append(StructureChunk(
                        text=current_text.strip(),
                        heading_level=section.heading_level,
                        heading_text=section.heading_text,
                        section_path=section.section_path,
                        chunk_type=section.element_type,
                    ))
                    # Add overlap from end of previous chunk
                    overlap = current_text[-self.overlap_size:] if len(current_text) > self.overlap_size else ""
                    current_text = overlap + " " + sentence
                else:
                    current_text = sentence
            else:
                current_text = current_text + " " + sentence if current_text else sentence
        
        if current_text.strip():
            chunks.append(StructureChunk(
                text=current_text.strip(),
                heading_level=section.heading_level,
                heading_text=section.heading_text,
                section_path=section.section_path,
                chunk_type=section.element_type,
            ))
        
        return chunks if chunks else [self._section_to_chunk(section)]
    
    def _merge_sections(self, sections: list[MarkdownSection]) -> StructureChunk:
        """Merge multiple sections into one chunk"""
        if not sections:
            return StructureChunk(text="")
        
        first = sections[0]
        merged_text = "\n\n".join(s.text for s in sections)
        
        types = [s.element_type for s in sections]
        chunk_type = max(set(types), key=types.count)
        
        # Preserve table metadata
        table_headers: list[str] = []
        table_row_count = 0
        for s in sections:
            if s.element_type == "table" and s.table_headers:
                table_headers = s.table_headers
                table_row_count = s.table_row_count
                break
        
        return StructureChunk(
            text=merged_text,
            heading_level=first.heading_level,
            heading_text=first.heading_text,
            parent_heading=self._get_parent_heading(first),
            section_path=first.section_path,
            chunk_type=chunk_type,
            table_headers=table_headers,
            table_row_count=table_row_count,
        )
    
    def _section_to_chunk(self, section: MarkdownSection) -> StructureChunk:
        """Convert a single section to chunk"""
        return StructureChunk(
            text=section.text,
            heading_level=section.heading_level,
            heading_text=section.heading_text,
            parent_heading=self._get_parent_heading(section),
            section_path=section.section_path,
            chunk_type=section.element_type,
            table_headers=section.table_headers,
            table_row_count=section.table_row_count,
        )
    
    def _get_parent_heading(self, section: MarkdownSection) -> str:
        """Extract parent heading"""
        path = section.section_path
        if len(path) >= 2:
            return path[-2]
        return ""


def parse_markdown_with_frontmatter(file_path: Path) -> tuple[dict, str]:
    """Parse markdown file with YAML front matter"""
    text = file_path.read_text(encoding="utf-8")
    
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1])
                content = parts[2].strip()
                return metadata or {}, content
            except yaml.YAMLError:
                pass
    
    return {}, text


def generate_chunk_id(source_file: str, chunk_index: int, chunk_text: str) -> str:
    """Generate unique chunk ID"""
    hash_input = f"{source_file}:{chunk_index}:{chunk_text[:100]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def generate_content_hash(text: str) -> str:
    """Generate content hash"""
    return hashlib.sha256(text.encode()).hexdigest()


def generate_source_hash(content: str, metadata: dict) -> str:
    """Generate source file hash"""
    hash_input = content + json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:32]


def load_existing_parquet(file_path: Path) -> tuple[pd.DataFrame | None, dict]:
    """Load existing parquet file"""
    if not file_path.exists():
        return None, {}
    
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        file_meta = table.schema.metadata or {}
        metadata = {
            "version": int(file_meta.get(b"version", b"0")),
            "source_hash": file_meta.get(b"source_hash", b"").decode(),
            "created_at": file_meta.get(b"created_at", b"").decode(),
            "updated_at": file_meta.get(b"updated_at", b"").decode(),
        }
        return df, metadata
    except Exception as e:
        print(f"⚠️ Failed to load existing file: {e}")
        return None, {}


def check_needs_update(existing_meta: dict, new_source_hash: str) -> bool:
    """Check if update is needed"""
    if not existing_meta:
        return True
    return existing_meta.get("source_hash") != new_source_hash


def merge_chunks(
    existing_df: pd.DataFrame | None,
    new_records: list[dict],
    existing_meta: dict,
) -> tuple[list[dict], dict]:
    """Incremental update merge"""
    stats = {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0}
    
    if existing_df is None or existing_df.empty:
        stats["added"] = len(new_records)
        return new_records, stats
    
    existing_hashes = set(existing_df["content_hash"].tolist()) if "content_hash" in existing_df.columns else set()
    existing_chunk_ids = set(existing_df["chunk_id"].tolist())
    
    merged = []
    new_hashes = set()
    new_chunk_ids = set()
    
    for record in new_records:
        content_hash = record["content_hash"]
        chunk_id = record["chunk_id"]
        new_hashes.add(content_hash)
        new_chunk_ids.add(chunk_id)
        
        if content_hash in existing_hashes:
            existing_row = existing_df[existing_df["content_hash"] == content_hash].iloc[0]
            record["version"] = int(existing_row["version"])
            record["created_at"] = existing_row["created_at"]
            stats["unchanged"] += 1
        else:
            if chunk_id in existing_chunk_ids:
                existing_row = existing_df[existing_df["chunk_id"] == chunk_id].iloc[0]
                record["version"] = int(existing_row["version"]) + 1
                record["created_at"] = existing_row["created_at"]
                stats["updated"] += 1
            else:
                stats["added"] += 1
        
        merged.append(record)
    
    deleted_chunk_ids = existing_chunk_ids - new_chunk_ids
    stats["deleted"] = len(deleted_chunk_ids)
    
    return merged, stats


def process_documents(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    min_chunk_size: int = MIN_CHUNK_SIZE,
):
    """Main document processing function"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"⚠️ Input directory does not exist: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_files = list(input_dir.glob("*.md"))
    
    if not md_files:
        print(f"⚠️ No markdown files to process: {input_dir}")
        return
    
    print(f"\n📚 Documents to process: {len(md_files)}")
    print("=" * 50)
    
    # Initialize chunker
    chunker = StructureChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
    )
    
    success_count = 0
    total_chunks = 0
    
    for i, md_file in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] Processing: {md_file.name}")
        
        try:
            # Parse markdown
            metadata, content = parse_markdown_with_frontmatter(md_file)
            print(f"   📖 Document length: {len(content)} characters")
            
            # Calculate source hash
            source_hash = generate_source_hash(content, metadata)
            output_file = output_dir / f"{md_file.stem}.parquet"
            
            # Check existing file
            existing_df, existing_meta = load_existing_parquet(output_file)
            
            # Check if update needed
            if not check_needs_update(existing_meta, source_hash):
                print(f"   ⏭️ No changes, skipping")
                success_count += 1
                if existing_df is not None:
                    total_chunks += len(existing_df)
                continue
            
            # Structure-based chunking
            print("   🔍 Structure-based chunking...")
            chunks = chunker.chunk_document(content, metadata)
            print(f"   ✓ Generated chunks: {len(chunks)}")
            
            now = datetime.now().isoformat()
            new_version = existing_meta.get("version", 0) + 1
            created_at = existing_meta.get("created_at") or now
            
            # Build records
            records = []
            for idx, chunk in enumerate(chunks):
                chunk_id = generate_chunk_id(md_file.name, idx, chunk.text)
                content_hash = generate_content_hash(chunk.text)
                
                records.append({
                    "chunk_id": chunk_id,
                    "content_hash": content_hash,
                    "source_file": md_file.name,
                    "chunk_index": idx,
                    "chunk_text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    # Hierarchy info
                    "heading_level": chunk.heading_level,
                    "heading_text": chunk.heading_text,
                    "parent_heading": chunk.parent_heading,
                    "section_path": chunk.section_path,
                    # Table metadata
                    "table_headers": chunk.table_headers,
                    "table_row_count": chunk.table_row_count,
                    # Metadata
                    "domain": metadata.get("domain", ""),
                    "sub_domain": metadata.get("sub_domain", ""),
                    "keywords": json.dumps(metadata.get("keywords", []), ensure_ascii=False),
                    "language": metadata.get("language", ""),
                    "content_type": metadata.get("content_type", ""),
                    # Version
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                })
            
            # Incremental update merge
            merged_records, update_stats = merge_chunks(existing_df, records, existing_meta)
            
            if existing_df is not None:
                print(f"   📊 Incremental update: added {update_stats['added']}, updated {update_stats['updated']}, "
                      f"unchanged {update_stats['unchanged']}, deleted {update_stats['deleted']}")
            
            # Save as Parquet
            df = pd.DataFrame(merged_records)
            
            schema = pa.schema([
                ("chunk_id", pa.string()),
                ("content_hash", pa.string()),
                ("source_file", pa.string()),
                ("chunk_index", pa.int32()),
                ("chunk_text", pa.string()),
                ("chunk_type", pa.string()),
                # Hierarchy
                ("heading_level", pa.int32()),
                ("heading_text", pa.string()),
                ("parent_heading", pa.string()),
                ("section_path", pa.list_(pa.string())),
                # Table metadata
                ("table_headers", pa.list_(pa.string())),
                ("table_row_count", pa.int32()),
                # Metadata
                ("domain", pa.string()),
                ("sub_domain", pa.string()),
                ("keywords", pa.string()),
                ("language", pa.string()),
                ("content_type", pa.string()),
                # Version
                ("version", pa.int32()),
                ("created_at", pa.string()),
                ("updated_at", pa.string()),
            ])
            
            file_metadata = {
                b"version": str(new_version).encode(),
                b"source_hash": source_hash.encode(),
                b"created_at": created_at.encode(),
                b"updated_at": now.encode(),
                b"schema_version": b"2.0",
                b"chunking_method": b"structure_based",
            }
            
            table = pa.Table.from_pandas(df, schema=schema)
            table = table.replace_schema_metadata(file_metadata)
            
            pq.write_table(
                table,
                output_file,
                compression="zstd",
                compression_level=3,
            )
            
            print(f"   💾 Saved: {output_file.name} (v{new_version}, zstd compression)")
            
            success_count += 1
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"✅ Complete: {success_count}/{len(md_files)} documents processed")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"📁 Output location: {output_dir}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split markdown documents into structure-based chunks and save as parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=MAX_CHUNK_SIZE,
        help=f"Maximum chunk size in characters (default: {MAX_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=MIN_CHUNK_SIZE,
        help=f"Minimum chunk size in characters (default: {MIN_CHUNK_SIZE})",
    )
    
    args = parser.parse_args()
    
    process_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        exit(130)
