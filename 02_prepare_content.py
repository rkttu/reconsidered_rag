"""
02_prepare_content.py
입력 문서에서 메타데이터를 추출하고 YAML front matter를 추가하는 모듈

특징:
- Microsoft markitdown을 활용한 다양한 문서 형식 지원
  (Word, Excel, PowerPoint, PDF, HTML, 이미지, 오디오 등)
- 마크다운 구조 분석을 통한 메타데이터 추출
- 키워드 자동 추출 (헤딩, 볼드, 링크 텍스트 등)
- 언어 감지 지원
- YAML front matter 생성
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
from langdetect import detect, LangDetectException
from markitdown import MarkItDown, UnsupportedFormatException


# 디렉터리 설정
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_docs"
OUTPUT_DIR = BASE_DIR / "prepared_contents"

# MarkItDown 지원 파일 확장자
SUPPORTED_EXTENSIONS = {
    # 문서
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    # 웹/텍스트
    ".html", ".htm", ".xml", ".json", ".csv",
    # 마크다운/텍스트
    ".md", ".markdown", ".txt", ".rst",
    # 이미지 (EXIF/OCR)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff",
    # 오디오 (음성 인식)
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    # 비디오 (자막 추출)
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    # 코드/기타
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs",
    ".ipynb",  # Jupyter Notebook
    ".zip",  # Archive (내부 파일 처리)
}

# MarkItDown 인스턴스 (싱글톤)
_markitdown_instance: Optional[MarkItDown] = None


def get_markitdown() -> MarkItDown:
    """MarkItDown 싱글톤 인스턴스 반환"""
    global _markitdown_instance
    if _markitdown_instance is None:
        _markitdown_instance = MarkItDown()
    return _markitdown_instance


def is_supported_file(file_path: Path) -> bool:
    """파일이 지원되는 형식인지 확인"""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def convert_to_markdown(file_path: Path) -> tuple[str, str]:
    """
    파일을 마크다운으로 변환

    Args:
        file_path: 변환할 파일 경로

    Returns:
        (마크다운 콘텐츠, 원본 파일 형식) 튜플

    Raises:
        UnsupportedFormatException: 지원하지 않는 형식
        Exception: 변환 실패
    """
    suffix = file_path.suffix.lower()

    # 이미 마크다운인 경우
    if suffix in {".md", ".markdown"}:
        content = file_path.read_text(encoding="utf-8")
        return content, "markdown"

    # 텍스트 파일인 경우
    if suffix == ".txt":
        content = file_path.read_text(encoding="utf-8")
        return content, "plaintext"

    # MarkItDown으로 변환
    md = get_markitdown()
    result = md.convert(str(file_path))

    if result.text_content:
        return result.text_content, suffix.lstrip(".")

    raise ValueError(f"변환 결과가 비어있습니다: {file_path}")


def detect_language(text: str) -> str:
    """텍스트 언어 감지 (langdetect 라이브러리 사용)"""
    if not text or len(text.strip()) < 10:
        return "unknown"
    
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def extract_title(content: str) -> str:
    """첫 번째 헤딩을 제목으로 추출"""
    # H1 찾기
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # H2 찾기
    h2_match = re.search(r'^##\s+(.+)$', content, re.MULTILINE)
    if h2_match:
        return h2_match.group(1).strip()
    
    # 첫 줄
    first_line = content.strip().split('\n')[0].strip()
    if first_line:
        return first_line[:100]
    
    return "Untitled"


def extract_keywords(content: str, max_keywords: int = 10) -> list[str]:
    """
    콘텐츠에서 키워드 추출
    
    - 헤딩 텍스트
    - 볼드/이탤릭 텍스트
    - 코드 블록 언어
    - 링크 텍스트
    """
    keywords = set()
    
    # 헤딩 추출
    headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
    for h in headings:
        # 특수문자 제거 후 단어 추출
        words = re.findall(r'\b[\w가-힣]{2,}\b', h)
        keywords.update(words)
    
    # 볼드 텍스트 추출 (**text** 또는 __text__)
    bold_texts = re.findall(r'\*\*(.+?)\*\*|__(.+?)__', content)
    for match in bold_texts:
        text = match[0] or match[1]
        words = re.findall(r'\b[\w가-힣]{2,}\b', text)
        keywords.update(words)
    
    # 코드 블록 언어 추출
    code_langs = re.findall(r'^```(\w+)', content, re.MULTILINE)
    keywords.update(code_langs)
    
    # 링크 텍스트 추출
    link_texts = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
    for lt in link_texts:
        if len(lt) > 2 and len(lt) < 50:
            keywords.add(lt.strip())
    
    # 불용어 제거 (간단한 목록)
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'and', 'or', 'but', 'if',
        'then', 'else', 'when', 'where', 'how', 'what', 'why',
        '이', '그', '저', '것', '수', '등', '및', '또는', '의', '를', '을',
    }
    
    filtered = [kw for kw in keywords if kw.lower() not in stopwords]
    
    # 길이 기준 정렬 후 상위 N개 반환
    sorted_kw = sorted(filtered, key=lambda x: len(x), reverse=True)
    
    return sorted_kw[:max_keywords]


def detect_content_type(content: str) -> str:
    """콘텐츠 타입 감지"""
    # 코드 블록 비율
    code_blocks = re.findall(r'```[\s\S]*?```', content)
    code_length = sum(len(cb) for cb in code_blocks)
    code_ratio = code_length / len(content) if content else 0
    
    if code_ratio > 0.5:
        return "code"
    elif code_ratio > 0.2:
        return "tutorial"
    
    # 리스트 비율
    list_items = re.findall(r'^[-*]\s+', content, re.MULTILINE)
    numbered_items = re.findall(r'^\d+\.\s+', content, re.MULTILINE)
    list_count = len(list_items) + len(numbered_items)
    
    lines = content.count('\n') + 1
    if list_count > lines * 0.3:
        return "list"
    
    # 헤딩 수
    headings = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
    if len(headings) > 5:
        return "documentation"
    
    return "article"


def infer_domain(content: str, keywords: list[str]) -> tuple[str, str]:
    """
    도메인 및 서브도메인 추론
    
    Returns:
        (domain, sub_domain) 튜플
    """
    content_lower = content.lower()
    keywords_lower = [kw.lower() for kw in keywords]
    
    # 도메인 키워드 매핑
    domain_keywords = {
        "programming": [
            "python", "javascript", "typescript", "java", "rust", "go",
            "code", "function", "class", "api", "library", "framework",
            "코드", "함수", "프로그래밍", "개발",
        ],
        "machine-learning": [
            "model", "training", "neural", "deep learning", "ai", "ml",
            "embedding", "transformer", "bert", "gpt", "llm",
            "모델", "학습", "인공지능", "딥러닝", "임베딩",
        ],
        "data-science": [
            "data", "pandas", "numpy", "analysis", "visualization",
            "데이터", "분석", "시각화", "통계",
        ],
        "devops": [
            "docker", "kubernetes", "ci/cd", "deploy", "container",
            "aws", "azure", "gcp", "cloud",
            "배포", "컨테이너", "클라우드",
        ],
        "web": [
            "html", "css", "react", "vue", "angular", "frontend", "backend",
            "웹", "프론트엔드", "백엔드",
        ],
        "database": [
            "sql", "nosql", "postgresql", "mongodb", "redis",
            "데이터베이스", "쿼리",
        ],
    }
    
    domain_scores: dict[str, int] = {}
    
    for domain, domain_kws in domain_keywords.items():
        score = 0
        for kw in domain_kws:
            if kw in content_lower:
                score += content_lower.count(kw)
            if kw in keywords_lower:
                score += 5  # 키워드에 있으면 가중치
        if score > 0:
            domain_scores[domain] = score
    
    if not domain_scores:
        return "general", ""
    
    # 최고 점수 도메인
    top_domain = max(domain_scores, key=lambda x: domain_scores[x])
    
    # 서브도메인은 해당 도메인의 키워드 중 가장 많이 나온 것
    sub_domain = ""
    if top_domain in domain_keywords:
        sub_counts = {}
        for kw in domain_keywords[top_domain]:
            count = content_lower.count(kw)
            if count > 0:
                sub_counts[kw] = count
        if sub_counts:
            sub_domain = max(sub_counts, key=lambda x: sub_counts[x])
    
    return top_domain, sub_domain


def create_summary(content: str, max_length: int = 300) -> str:
    """첫 번째 문단을 요약으로 사용"""
    # YAML front matter 제거
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]
    
    # 헤딩 제거
    lines = content.strip().split('\n')
    paragraphs = []
    current = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            if current:
                paragraphs.append(' '.join(current))
                current = []
            continue
        if stripped == '':
            if current:
                paragraphs.append(' '.join(current))
                current = []
        else:
            current.append(stripped)
    
    if current:
        paragraphs.append(' '.join(current))
    
    # 첫 번째 의미있는 문단
    for p in paragraphs:
        # 마크다운 문법 제거
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', p)  # 링크
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)  # 볼드
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)  # 이탤릭
        clean = re.sub(r'`([^`]+)`', r'\1', clean)  # 인라인 코드
        
        if len(clean) > 30:
            if len(clean) > max_length:
                return clean[:max_length] + "..."
            return clean
    
    return ""


def prepare_document(
    input_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    문서에 YAML front matter 메타데이터 추가

    Args:
        input_path: 입력 파일 경로 (다양한 형식 지원)
        output_path: 출력 경로 (기본값: OUTPUT_DIR)

    Returns:
        생성된 파일 경로
    """
    # 파일을 마크다운으로 변환
    content, source_format = convert_to_markdown(input_path)

    # 기존 front matter가 있으면 제거
    original_content = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            original_content = parts[2].strip()

    # 메타데이터 추출
    title = extract_title(original_content)
    keywords = extract_keywords(original_content)
    language = detect_language(original_content)
    content_type = detect_content_type(original_content)
    domain, sub_domain = infer_domain(original_content, keywords)
    summary = create_summary(original_content)

    # YAML front matter 생성
    metadata = {
        "title": title,
        "domain": domain,
        "sub_domain": sub_domain,
        "keywords": keywords,
        "summary": summary,
        "language": language,
        "content_type": content_type,
        "source_file": input_path.name,
        "source_format": source_format,
        "prepared_at": datetime.now().isoformat(),
    }

    # 출력 경로 결정 (항상 .md 확장자)
    if output_path is None:
        output_path = OUTPUT_DIR / input_path.with_suffix('.md').name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 새 콘텐츠 작성
    yaml_header = yaml.dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False)
    new_content = f"---\n{yaml_header}---\n\n{original_content}"

    output_path.write_text(new_content, encoding="utf-8")

    return output_path


def process_all_documents(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """
    모든 입력 문서 처리 (다양한 형식 지원)

    Args:
        input_dir: 입력 디렉터리
        output_dir: 출력 디렉터리

    Returns:
        생성된 파일 경로 리스트
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"⚠️ 입력 디렉터리가 없습니다: {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # 지원되는 모든 파일 수집
    all_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and is_supported_file(f)
    ]

    if not all_files:
        print(f"⚠️ 처리할 파일이 없습니다: {input_dir}")
        print(f"   지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return []

    # 파일 형식별 통계
    format_counts: dict[str, int] = {}
    for f in all_files:
        ext = f.suffix.lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1

    print(f"\n📚 처리할 문서: {len(all_files)}개")
    print(f"   형식별: {', '.join(f'{ext}({cnt})' for ext, cnt in sorted(format_counts.items()))}")
    print("=" * 50)

    results = []
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] {file_path.name}")

        try:
            output_path = output_dir / file_path.with_suffix('.md').name
            result = prepare_document(file_path, output_path)

            # 결과 확인
            content = result.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    meta = yaml.safe_load(parts[1])
                    print(f"   • 제목: {meta.get('title', 'N/A')[:40]}...")
                    print(f"   • 도메인: {meta.get('domain', 'N/A')}")
                    print(f"   • 원본 형식: {meta.get('source_format', 'N/A')}")
                    print(f"   • 키워드: {', '.join(meta.get('keywords', [])[:5])}")
                    print(f"   ✅ 저장: {result.name}")

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
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="입력 문서에 YAML front matter 메타데이터를 추가합니다."
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
    exit(main())
