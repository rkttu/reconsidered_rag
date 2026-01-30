"""
prepare_utils.py
Common utilities for 01_prepare_* scripts

This module provides shared functionality for all document preparation scripts:
- Azure service client creation
- Metadata extraction (title, keywords, domain, language, etc.)
- YAML front matter generation
- Language detection
- Content type inference

All 01_prepare_* scripts should import from this module to ensure consistency.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

import yaml
from dotenv import load_dotenv
from langdetect import detect, LangDetectException


# Load .env file
load_dotenv()

# Directory configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_docs"
OUTPUT_DIR = BASE_DIR / "prepared_contents"


# =============================================================================
# Unicode Constants (encoding-safe)
# =============================================================================

# Circled Numbers - ① ~ ⑳
CIRCLED_NUMBERS = [
    '\u2460', '\u2461', '\u2462', '\u2463', '\u2464',
    '\u2465', '\u2466', '\u2467', '\u2468', '\u2469',
    '\u246A', '\u246B', '\u246C', '\u246D', '\u246E',
    '\u246F', '\u2470', '\u2471', '\u2472', '\u2473',
]

# Korean article title brackets
LEFT_BLACK_LENTICULAR_BRACKET = '\u3010'  # 【
RIGHT_BLACK_LENTICULAR_BRACKET = '\u3011'  # 】


# =============================================================================
# Azure Service Configuration
# =============================================================================

def get_azure_document_intelligence_client() -> Optional[Any]:
    """Create Azure Document Intelligence client"""
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    auth_method = os.getenv("AZURE_AUTH_METHOD", "key")

    if not endpoint:
        return None

    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient

        if auth_method == "default":
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
        else:
            from azure.core.credentials import AzureKeyCredential
            if not key:
                print("⚠️ AZURE_DOCUMENT_INTELLIGENCE_KEY is not set.")
                return None
            credential = AzureKeyCredential(key)

        return DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=credential,
        )
    except ImportError:
        print("⚠️ azure-ai-documentintelligence package is not installed.")
        return None
    except Exception as e:
        print(f"⚠️ Failed to create Document Intelligence client: {e}")
        return None


def get_azure_openai_client() -> Optional[Any]:
    """
    Create Azure OpenAI client (GPT-4o Vision)
    
    Supports both Azure OpenAI and Azure AI Foundry (OpenAI-compatible) endpoints.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not key:
        return None

    try:
        # Check if it's Azure AI Foundry OpenAI-compatible endpoint
        if "cognitiveservices.azure.com" in endpoint or "/v1" in endpoint:
            from openai import OpenAI  # type: ignore[import-untyped]
            
            if not endpoint.rstrip("/").endswith("/v1"):
                endpoint = endpoint.rstrip("/") + "/openai/v1/"
            
            return OpenAI(
                base_url=endpoint,
                api_key=key,
            )
        else:
            from openai import AzureOpenAI  # type: ignore[import-untyped]

            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            )
    except ImportError:
        print("⚠️ openai package is not installed.")
        return None
    except Exception as e:
        print(f"⚠️ Failed to create Azure OpenAI client: {e}")
        return None


def get_azure_speech_config() -> Optional[tuple[str, str]]:
    """Return Azure Speech configuration"""
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")

    if key and region:
        return key, region
    return None


# =============================================================================
# Language Detection
# =============================================================================

def detect_language(text: str) -> str:
    """Detect text language using langdetect"""
    if not text or len(text.strip()) < 10:
        return "unknown"
    
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# =============================================================================
# Metadata Extraction
# =============================================================================

def extract_title(content: str) -> str:
    """Extract first heading as title"""
    # Find H1
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Find H2
    h2_match = re.search(r'^##\s+(.+)$', content, re.MULTILINE)
    if h2_match:
        return h2_match.group(1).strip()
    
    # First line
    first_line = content.strip().split('\n')[0].strip()
    if first_line:
        return first_line[:100]
    
    return "Untitled"


def extract_keywords(content: str, max_keywords: int = 10) -> list[str]:
    """
    Extract keywords from content
    
    Sources:
    - Heading texts
    - Bold/italic texts
    - Code block languages
    - Link texts
    """
    keywords = set()
    
    # Extract headings
    headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
    for h in headings:
        words = re.findall(r'\b[\w가-힣]{2,}\b', h)
        keywords.update(words)
    
    # Extract bold text
    bold_texts = re.findall(r'\*\*(.+?)\*\*|__(.+?)__', content)
    for match in bold_texts:
        text = match[0] or match[1]
        words = re.findall(r'\b[\w가-힣]{2,}\b', text)
        keywords.update(words)
    
    # Extract code block languages
    code_langs = re.findall(r'^```(\w+)', content, re.MULTILINE)
    keywords.update(code_langs)
    
    # Extract link texts
    link_texts = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
    for lt in link_texts:
        if len(lt) > 2 and len(lt) < 50:
            keywords.add(lt.strip())
    
    # Remove stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'and', 'or', 'but', 'if',
        'then', 'else', 'when', 'where', 'how', 'what', 'why',
        '이', '그', '저', '것', '수', '등', '및', '또는', '의', '를', '을',
    }
    
    filtered = [kw for kw in keywords if kw.lower() not in stopwords]
    sorted_kw = sorted(filtered, key=lambda x: len(x), reverse=True)
    
    return sorted_kw[:max_keywords]


def detect_content_type(content: str) -> str:
    """Detect content type"""
    # Code block ratio
    code_blocks = re.findall(r'```[\s\S]*?```', content)
    code_length = sum(len(cb) for cb in code_blocks)
    code_ratio = code_length / len(content) if content else 0
    
    if code_ratio > 0.5:
        return "code"
    elif code_ratio > 0.2:
        return "tutorial"
    
    # List ratio
    list_items = re.findall(r'^[-*]\s+', content, re.MULTILINE)
    numbered_items = re.findall(r'^\d+\.\s+', content, re.MULTILINE)
    list_count = len(list_items) + len(numbered_items)
    
    lines = content.count('\n') + 1
    if list_count > lines * 0.3:
        return "list"
    
    # Heading count
    headings = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
    if len(headings) > 5:
        return "documentation"
    
    return "article"


def infer_domain(content: str, keywords: list[str]) -> tuple[str, str]:
    """
    Infer domain and sub-domain
    
    Returns:
        (domain, sub_domain) tuple
    """
    content_lower = content.lower()
    keywords_lower = [kw.lower() for kw in keywords]
    
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
                score += 5
        if score > 0:
            domain_scores[domain] = score
    
    if not domain_scores:
        return "general", ""
    
    top_domain = max(domain_scores, key=lambda x: domain_scores[x])
    
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
    """Use first paragraph as summary"""
    # Remove YAML front matter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]
    
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
    
    for p in paragraphs:
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', p)
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        
        if len(clean) > 30:
            if len(clean) > max_length:
                return clean[:max_length] + "..."
            return clean
    
    return ""


# =============================================================================
# YAML Front Matter
# =============================================================================

def create_front_matter(
    content: str,
    source_file: str,
    source_format: str,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create YAML front matter metadata from content
    
    Args:
        content: Markdown content (without existing front matter)
        source_file: Original file name
        source_format: Original file format
        extra_metadata: Additional metadata to include
    
    Returns:
        Metadata dictionary
    """
    title = extract_title(content)
    keywords = extract_keywords(content)
    language = detect_language(content)
    content_type = detect_content_type(content)
    domain, sub_domain = infer_domain(content, keywords)
    summary = create_summary(content)

    metadata: dict[str, Any] = {
        "title": title,
        "domain": domain,
        "sub_domain": sub_domain,
        "keywords": keywords,
        "summary": summary,
        "language": language,
        "content_type": content_type,
        "source_file": source_file,
        "source_format": source_format,
        "prepared_at": datetime.now().isoformat(),
    }
    
    if extra_metadata:
        metadata.update(extra_metadata)
    
    return metadata


def strip_existing_front_matter(content: str) -> str:
    """Remove existing YAML front matter from content"""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content


def write_with_front_matter(
    output_path: Path,
    content: str,
    metadata: dict[str, Any],
) -> Path:
    """
    Write content with YAML front matter
    
    Args:
        output_path: Output file path
        content: Markdown content (without front matter)
        metadata: Metadata dictionary
    
    Returns:
        Written file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    yaml_header = yaml.dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    new_content = f"---\n{yaml_header}---\n\n{content}"
    
    output_path.write_text(new_content, encoding="utf-8")
    
    return output_path


def print_processing_result(output_path: Path, metadata: dict[str, Any]) -> None:
    """Print processing result summary"""
    print(f"   • 제목: {metadata.get('title', 'N/A')[:40]}...")
    print(f"   • 도메인: {metadata.get('domain', 'N/A')}")
    print(f"   • 원본 형식: {metadata.get('source_format', 'N/A')}")
    print(f"   • 키워드: {', '.join(metadata.get('keywords', [])[:5])}")
    print(f"   ✅ 저장: {output_path.name}")
