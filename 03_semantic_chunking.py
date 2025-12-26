"""
03_semantic_chunking.py
prepared_contents의 마크다운 파일을 BGE-M3 임베딩 모델을 사용하여
시맨틱 청킹으로 분할하고 parquet 파일로 저장하는 모듈

특징:
- Markdown 구조 파싱 (heading hierarchy 보존)
- BGE-M3 임베딩 기반 시맨틱 유사도 청킹
- zstd 압축 및 증분 업데이트 지원
"""

import re
import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import mistune
import torch
from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]


def get_device_info() -> tuple[str, bool]:
    """
    사용 가능한 디바이스 및 FP16 지원 여부 반환

    Returns:
        (device_name, use_fp16) 튜플
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return device_name, True
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3)
        return "Apple MPS", False  # MPS는 FP16이 제한적
    else:
        return "CPU", False


# 디렉터리 설정
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "prepared_contents"
OUTPUT_DIR = BASE_DIR / "chunked_data"

# 시맨틱 청킹 설정
SIMILARITY_THRESHOLD = 0.5  # 유사도 임계값 (낮을수록 더 많은 청크 분할)
MIN_CHUNK_SIZE = 50  # 최소 청크 크기 (문자 수)
MAX_CHUNK_SIZE = 1500  # 최대 청크 크기 (문자 수)


@dataclass
class MarkdownSection:
    """마크다운 섹션 정보"""
    text: str
    heading_level: int = 0  # 0 = 일반 텍스트, 1-6 = 헤딩 레벨
    heading_text: str = ""
    section_path: list[str] = field(default_factory=list)  # 계층 경로 배열
    element_type: str = "paragraph"  # header, paragraph, list, code, blockquote, table
    line_start: int = 0
    line_end: int = 0
    # 표 전용 메타데이터
    table_headers: list[str] = field(default_factory=list)
    table_row_count: int = 0


@dataclass
class SemanticChunk:
    """시맨틱 청크 정보"""
    text: str
    heading_level: int = 0
    heading_text: str = ""
    parent_heading: str = ""
    section_path: list[str] = field(default_factory=list)  # 계층 경로 배열
    chunk_type: str = "paragraph"
    start_line: int = 0
    end_line: int = 0
    # 표 전용 메타데이터
    table_headers: list[str] = field(default_factory=list)
    table_row_count: int = 0


class MarkdownParser:
    """Mistune 기반 마크다운 파서"""
    
    def __init__(self):
        self.sections: list[MarkdownSection] = []
        self.current_headings: dict[int, str] = {}  # level -> heading text
    
    def parse(self, markdown_text: str) -> list[MarkdownSection]:
        """
        마크다운을 파싱하여 섹션 리스트 반환
        
        Args:
            markdown_text: 마크다운 텍스트
            
        Returns:
            MarkdownSection 리스트
        """
        self.sections = []
        self.current_headings = {}
        
        # mistune으로 AST 파싱 (table 플러그인 활성화)
        md = mistune.create_markdown(renderer=None, plugins=['table'])
        tokens = md(markdown_text)
        
        if tokens is None:
            tokens = []
        
        # 라인 번호 추적을 위해 원본 텍스트 분할
        lines = markdown_text.split('\n')
        current_line = 0
        
        for token in tokens:
            if isinstance(token, dict):
                self._process_token(token, lines, current_line)
        
        # 토큰 기반 파싱이 비어있으면 라인 기반 파싱으로 폴백
        if not self.sections:
            self._fallback_parse(markdown_text)
        
        return self.sections
    
    def _process_token(self, token: dict, lines: list[str], line_offset: int) -> None:
        """토큰 처리"""
        token_type = token.get('type', '')
        
        if token_type == 'heading':
            level = token.get('attrs', {}).get('level', 1)
            # children에서 텍스트 추출
            children = token.get('children', [])
            text = self._extract_text_from_children(children)
            
            # 현재 헤딩 업데이트
            self.current_headings[level] = text
            # 하위 레벨 초기화
            for l in range(level + 1, 7):
                self.current_headings.pop(l, None)
            
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
            # 표 처리: 전체 표를 하나의 청크로 유지
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
        """children 토큰에서 텍스트 추출"""
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
        """리스트 아이템에서 텍스트 추출"""
        texts = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                children = item.get('children', [])
                item_text = self._extract_text_from_children(children)
                texts.append(f"- {item_text}")
        return '\n'.join(texts)
    
    def _extract_table(self, token: dict) -> tuple[str, list[str], int]:
        """
        표 토큰에서 텍스트, 헤더, 행 수 추출
        
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
                # 표 헤더 추출 - table_head가 직접 table_cell을 포함
                head_cells = child.get('children', [])
                for cell in head_cells:
                    if isinstance(cell, dict) and cell.get('type') == 'table_cell':
                        cell_text = self._extract_text_from_children(
                            cell.get('children', [])
                        )
                        headers.append(cell_text.strip())
            
            elif child_type == 'table_body':
                # 표 본문 추출
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
        
        # 마크다운 표 형식으로 재구성
        md_lines = []
        if headers:
            md_lines.append('| ' + ' | '.join(headers) + ' |')
            md_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        for row in rows:
            # 헤더 수에 맞춰 패딩
            padded_row = row + [''] * (len(headers) - len(row)) if headers else row
            md_lines.append('| ' + ' | '.join(padded_row) + ' |')
        
        table_text = '\n'.join(md_lines)
        return table_text, headers, len(rows)
    
    def _build_section_path(self) -> list[str]:
        """현재 섹션 경로를 배열로 생성"""
        parts = []
        for level in sorted(self.current_headings.keys()):
            parts.append(self.current_headings[level])
        return parts
    
    def _get_current_heading(self) -> str:
        """현재 가장 깊은 헤딩 반환"""
        if self.current_headings:
            max_level = max(self.current_headings.keys())
            return self.current_headings[max_level]
        return ""
    
    def _fallback_parse(self, markdown_text: str) -> None:
        """라인 기반 폴백 파싱"""
        lines = markdown_text.split('\n')
        current_headings: dict[int, str] = {}
        current_paragraph: list[str] = []
        
        def flush_paragraph():
            if current_paragraph:
                text = '\n'.join(current_paragraph).strip()
                if text:
                    section_path = [
                        h for l, h in sorted(current_headings.items())
                    ]
                    current_heading = current_headings.get(
                        max(current_headings.keys()) if current_headings else 0, ""
                    )
                    
                    # 리스트인지 확인
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
        
        # 표 감지 및 처리 함수
        def is_table_line(line: str) -> bool:
            """표 라인인지 확인 (| 로 시작하고 끝남)"""
            s = line.strip()
            return s.startswith('|') and s.endswith('|')
        
        def is_separator_line(line: str) -> bool:
            """표 구분선인지 확인 (|---|---|)"""
            s = line.strip()
            if not (s.startswith('|') and s.endswith('|')):
                return False
            # 중간에 ---가 포함되어 있는지
            return bool(re.match(r'^\|[\s\-:|]+\|$', s))
        
        def flush_table(table_lines: list[str], start_idx: int):
            """표를 섹션으로 추가"""
            if not table_lines:
                return
            
            table_text = '\n'.join(table_lines)
            headers: list[str] = []
            row_count = 0
            
            # 첫 줄에서 헤더 추출
            if table_lines:
                first_line = table_lines[0].strip()
                if first_line.startswith('|') and first_line.endswith('|'):
                    cells = [c.strip() for c in first_line[1:-1].split('|')]
                    headers = [c for c in cells if c]
            
            # 구분선 제외하고 데이터 행 수 계산
            for tl in table_lines[2:]:  # 헤더, 구분선 제외
                if is_table_line(tl) and not is_separator_line(tl):
                    row_count += 1
            
            section_path = [
                h for l, h in sorted(current_headings.items())
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
            
            # 헤딩 감지
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if heading_match:
                flush_paragraph()
                
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                # 헤딩 업데이트
                current_headings[level] = heading_text
                for l in list(current_headings.keys()):
                    if l > level:
                        del current_headings[l]
                
                section_path = [
                    h for l, h in sorted(current_headings.items())
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
            
            # 표 감지 (| 로 시작하는 라인)
            elif is_table_line(stripped):
                flush_paragraph()
                
                # 연속된 표 라인 수집
                table_lines = [line]
                table_start = i
                i += 1
                
                while i < len(lines) and is_table_line(lines[i].strip()):
                    table_lines.append(lines[i])
                    i += 1
                
                # 최소 2줄 이상이면 표로 처리 (헤더 + 구분선)
                if len(table_lines) >= 2:
                    flush_table(table_lines, table_start)
                else:
                    # 표가 아니면 일반 문단으로
                    current_paragraph.extend(table_lines)
            
            elif stripped == '':
                flush_paragraph()
                i += 1
            
            else:
                current_paragraph.append(line)
                i += 1
        
        flush_paragraph()


class SemanticChunker:
    """BGE-M3 기반 시맨틱 청킹"""
    
    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        min_chunk_size: int = MIN_CHUNK_SIZE,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.model: Any = None
        self.parser = MarkdownParser()
        self._load_model()
    
    def _load_model(self) -> None:
        """BGE-M3 모델 로드 (GPU 없으면 CPU 폴백)"""
        device_name, use_fp16 = get_device_info()

        if use_fp16:
            print(f"🔄 BGE-M3 모델 로딩 중... (GPU: {device_name}, FP16)")
        else:
            print(f"🔄 BGE-M3 모델 로딩 중... ({device_name}, FP32)")

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)
        print("✅ 모델 로딩 완료")
    
    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트의 임베딩 계산"""
        if not texts:
            return np.array([])
        
        result = self.model.encode(texts, batch_size=32)
        return result["dense_vecs"]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def chunk_document(self, content: str, metadata: dict) -> list[SemanticChunk]:
        """
        문서를 시맨틱 청크로 분할
        
        Args:
            content: 마크다운 문서 내용
            metadata: 문서 메타데이터
            
        Returns:
            SemanticChunk 리스트
        """
        # 1. 마크다운 파싱
        sections = self.parser.parse(content)
        
        if not sections:
            return [SemanticChunk(
                text=content,
                chunk_type="document",
            )]
        
        # 2. 섹션 텍스트 추출 및 임베딩
        section_texts = [s.text for s in sections]
        
        if len(section_texts) <= 1:
            # 섹션이 1개 이하면 그대로 반환
            return [
                SemanticChunk(
                    text=s.text,
                    heading_level=s.heading_level,
                    heading_text=s.heading_text,
                    parent_heading=self._get_parent_heading(s),
                    section_path=s.section_path,
                    chunk_type=s.element_type,
                )
                for s in sections
            ]
        
        embeddings = self._get_embeddings(section_texts)
        
        # 3. 유사도 기반 청킹
        chunks: list[SemanticChunk] = []
        current_sections: list[MarkdownSection] = [sections[0]]
        current_text_length = len(sections[0].text)
        
        for i in range(1, len(sections)):
            section = sections[i]
            prev_section = sections[i - 1]
            
            # 헤딩은 항상 새 청크 시작
            if section.heading_level > 0:
                # 이전 청크 저장
                if current_sections:
                    chunks.append(self._merge_sections(current_sections))
                current_sections = [section]
                current_text_length = len(section.text)
                continue
            
            # 유사도 계산
            similarity = self._cosine_similarity(embeddings[i], embeddings[i - 1])
            
            # 청크 분할 조건:
            # 1. 유사도가 임계값 미만
            # 2. 또는 최대 크기 초과
            should_split = (
                similarity < self.similarity_threshold
                or current_text_length + len(section.text) > self.max_chunk_size
            )
            
            if should_split and current_text_length >= self.min_chunk_size:
                # 이전 청크 저장
                chunks.append(self._merge_sections(current_sections))
                current_sections = [section]
                current_text_length = len(section.text)
            else:
                # 현재 청크에 추가
                current_sections.append(section)
                current_text_length += len(section.text)
        
        # 마지막 청크 저장
        if current_sections:
            chunks.append(self._merge_sections(current_sections))
        
        return chunks
    
    def _merge_sections(self, sections: list[MarkdownSection]) -> SemanticChunk:
        """여러 섹션을 하나의 청크로 병합"""
        if not sections:
            return SemanticChunk(text="")
        
        # 첫 번째 섹션의 메타데이터 사용
        first = sections[0]
        
        # 텍스트 병합
        merged_text = "\n\n".join(s.text for s in sections)
        
        # 타입 결정 (가장 많은 타입 또는 첫 번째)
        types = [s.element_type for s in sections]
        chunk_type = max(set(types), key=types.count)
        
        # 표인 경우 메타데이터 전달
        table_headers: list[str] = []
        table_row_count = 0
        for s in sections:
            if s.element_type == "table" and s.table_headers:
                table_headers = s.table_headers
                table_row_count = s.table_row_count
                break
        
        return SemanticChunk(
            text=merged_text,
            heading_level=first.heading_level,
            heading_text=first.heading_text,
            parent_heading=self._get_parent_heading(first),
            section_path=first.section_path,
            chunk_type=chunk_type,
            table_headers=table_headers,
            table_row_count=table_row_count,
        )
    
    def _get_parent_heading(self, section: MarkdownSection) -> str:
        """부모 헤딩 추출"""
        path = section.section_path
        if len(path) >= 2:
            return path[-2]
        return ""


def parse_markdown_with_frontmatter(file_path: Path) -> tuple[dict, str]:
    """YAML front matter가 있는 마크다운 파일 파싱"""
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
    """청크 고유 ID 생성"""
    hash_input = f"{source_file}:{chunk_index}:{chunk_text[:100]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def generate_content_hash(text: str) -> str:
    """콘텐츠 해시 생성"""
    return hashlib.sha256(text.encode()).hexdigest()


def generate_source_hash(content: str, metadata: dict) -> str:
    """소스 파일 전체 해시"""
    hash_input = content + json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:32]


def load_existing_parquet(file_path: Path) -> tuple[pd.DataFrame | None, dict]:
    """기존 parquet 파일 로드"""
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
        print(f"⚠️ 기존 파일 로드 실패: {e}")
        return None, {}


def check_needs_update(existing_meta: dict, new_source_hash: str) -> bool:
    """업데이트 필요 여부 확인"""
    if not existing_meta:
        return True
    return existing_meta.get("source_hash") != new_source_hash


def merge_chunks(
    existing_df: pd.DataFrame | None,
    new_records: list[dict],
    existing_meta: dict,
) -> tuple[list[dict], dict]:
    """증분 업데이트 병합"""
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
    similarity_threshold: float = SIMILARITY_THRESHOLD,
):
    """문서 처리 메인 함수"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"⚠️ 입력 디렉터리가 없습니다: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_files = list(input_dir.glob("*.md"))
    
    if not md_files:
        print(f"⚠️ 처리할 마크다운 파일이 없습니다: {input_dir}")
        return
    
    print(f"\n📚 처리할 문서: {len(md_files)}개")
    print("=" * 50)
    
    # 청커 로드
    chunker = SemanticChunker(similarity_threshold=similarity_threshold)
    
    success_count = 0
    total_chunks = 0
    
    for i, md_file in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] 처리 중: {md_file.name}")
        
        try:
            # 마크다운 파싱
            metadata, content = parse_markdown_with_frontmatter(md_file)
            print(f"   📖 문서 길이: {len(content)} 문자")
            
            # 소스 해시 계산
            source_hash = generate_source_hash(content, metadata)
            output_file = output_dir / f"{md_file.stem}.parquet"
            
            # 기존 파일 확인
            existing_df, existing_meta = load_existing_parquet(output_file)
            
            # 변경 여부 확인
            if not check_needs_update(existing_meta, source_hash):
                print(f"   ⏭️ 변경 없음, 스킵")
                success_count += 1
                if existing_df is not None:
                    total_chunks += len(existing_df)
                continue
            
            # 시맨틱 청킹
            print("   🔍 시맨틱 청킹 중...")
            chunks = chunker.chunk_document(content, metadata)
            print(f"   ✓ 생성된 청크: {len(chunks)}개")
            
            now = datetime.now().isoformat()
            new_version = existing_meta.get("version", 0) + 1
            created_at = existing_meta.get("created_at") or now
            
            # 레코드 구성
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
                    # Hierarchy 정보 (section_path는 배열)
                    "heading_level": chunk.heading_level,
                    "heading_text": chunk.heading_text,
                    "parent_heading": chunk.parent_heading,
                    "section_path": chunk.section_path,  # list[str]
                    # 표 메타데이터
                    "table_headers": chunk.table_headers,  # list[str]
                    "table_row_count": chunk.table_row_count,
                    # 메타데이터
                    "domain": metadata.get("domain", ""),
                    "sub_domain": metadata.get("sub_domain", ""),
                    "keywords": json.dumps(metadata.get("keywords", []), ensure_ascii=False),
                    "language": metadata.get("language", ""),
                    "content_type": metadata.get("content_type", ""),
                    # 버전 관리
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                })
            
            # 증분 업데이트 병합
            merged_records, update_stats = merge_chunks(existing_df, records, existing_meta)
            
            if existing_df is not None:
                print(f"   📊 증분 업데이트: 추가 {update_stats['added']}, 수정 {update_stats['updated']}, "
                      f"유지 {update_stats['unchanged']}, 삭제 {update_stats['deleted']}")
            
            # Parquet 저장
            df = pd.DataFrame(merged_records)
            
            schema = pa.schema([
                ("chunk_id", pa.string()),
                ("content_hash", pa.string()),
                ("source_file", pa.string()),
                ("chunk_index", pa.int32()),
                ("chunk_text", pa.string()),
                ("chunk_type", pa.string()),
                # Hierarchy (section_path는 배열)
                ("heading_level", pa.int32()),
                ("heading_text", pa.string()),
                ("parent_heading", pa.string()),
                ("section_path", pa.list_(pa.string())),  # 배열 타입
                # Table metadata
                ("table_headers", pa.list_(pa.string())),  # 표 컬럼 헤더
                ("table_row_count", pa.int32()),  # 표 행 수
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
                b"chunking_method": b"semantic_bge_m3",
            }
            
            table = pa.Table.from_pandas(df, schema=schema)
            table = table.replace_schema_metadata(file_metadata)
            
            pq.write_table(
                table,
                output_file,
                compression="zstd",
                compression_level=3,
            )
            
            print(f"   💾 저장: {output_file.name} (v{new_version}, zstd 압축)")
            
            success_count += 1
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"✅ 완료: {success_count}/{len(md_files)} 문서 처리됨")
    print(f"📊 총 청크 수: {total_chunks}개")
    print(f"📁 출력 위치: {output_dir}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="마크다운 문서를 시맨틱 청킹하여 parquet으로 저장합니다."
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
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"유사도 임계값 (기본값: {SIMILARITY_THRESHOLD})",
    )
    
    args = parser.parse_args()
    
    process_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[중단됨]")
        exit(130)
