"""
02_enrich_content.py
Content enrichment: expand bullet points and lists into full sentences

Features:
- Converts bullet points to complete sentences
- Adds contextual information to terse content
- Preserves original structure while improving embedding quality
- Falls back to simple copy if no API configured

Input: prepared_contents/
Output: enriched_contents/
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Directory configuration
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "prepared_contents"
OUTPUT_DIR = BASE_DIR / "enriched_contents"


# =============================================================================
# LLM Client Configuration
# =============================================================================

def get_enrichment_client() -> Optional[tuple[str, Any]]:
    """
    Get LLM client for content enrichment.
    
    Supports:
    - Microsoft Foundry (Azure AI Inference API)
    - Azure OpenAI
    
    Returns None if environment variables are not set.
    """
    endpoint = os.getenv("ENRICHMENT_ENDPOINT")
    api_key = os.getenv("ENRICHMENT_API_KEY")
    
    if not endpoint or not api_key:
        return None
    
    try:
        # Try Azure AI Inference SDK first (for Microsoft Foundry)
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        return ("azure-ai-inference", client)
    except ImportError:
        pass
    
    try:
        # Fallback to OpenAI SDK (for Azure OpenAI)
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.getenv("ENRICHMENT_API_VERSION", "2024-02-15-preview"),
        )
        return ("openai", client)
    except ImportError:
        print("⚠️ Neither azure-ai-inference nor openai package is installed.")
        print("   Install with: uv sync --extra enrich")
        return None
    except Exception as e:
        print(f"⚠️ Failed to create enrichment client: {e}")
        return None


def expand_content_with_llm(
    content: str,
    metadata: dict,
    client_info: tuple[str, Any],
    model: Optional[str] = None,
) -> str:
    """
    Expand terse content (bullet points, lists) into complete sentences.
    
    This improves embedding quality by making implicit context explicit.
    
    Args:
        content: Original markdown content (without front matter)
        metadata: Document metadata for context
        client_info: Tuple of (client_type, client)
        model: Model deployment name
    
    Returns:
        Expanded markdown content
    """
    client_type, client = client_info
    model = model or os.getenv("ENRICHMENT_MODEL", "gpt-4.1")
    
    # Get document context
    title = metadata.get("title", "Unknown Document")
    domain = metadata.get("domain", "general")
    
    # Truncate content to avoid token limits
    max_chars = 48000
    truncated = content[:max_chars] if len(content) > max_chars else content
    was_truncated = len(content) > max_chars
    
    system_prompt = """You are a technical writer improving documentation for semantic search.

Your task: Expand terse content (bullet points, numbered lists, short phrases) into complete, 
self-contained sentences while preserving all original information.

Rules:
1. Convert bullet points and lists into flowing paragraphs
2. Add implicit context that makes each section self-explanatory
3. Preserve all technical details, code blocks, and tables exactly
4. Keep headings intact (##, ###, etc.)
5. Maintain the original structure and order
6. Do NOT add new information not implied by the original
7. Do NOT remove any information
8. Write in the same language as the original

Example transformation:
BEFORE:
## Installation
- Python 3.11+
- Use uv package manager
- Run `uv sync`

AFTER:
## Installation

To install this project, you need Python version 3.11 or higher. This project uses the uv package manager for dependency management. After cloning the repository, run the `uv sync` command to install all required dependencies.

Return ONLY the improved markdown content, no explanations."""

    user_prompt = f"""Document: {title}
Domain: {domain}

Expand this content into complete sentences:

{truncated}"""

    try:
        if client_type == "azure-ai-inference":
            from azure.ai.inference.models import SystemMessage, UserMessage
            
            response = client.complete(
                model=model,
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt),
                ],
                max_tokens=8000,
                temperature=0.3,
            )
            result_text = response.choices[0].message.content
        else:
            # OpenAI SDK
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8000,
                temperature=0.3,
            )
            result_text = response.choices[0].message.content
        
        # If content was truncated, append the remaining part
        if was_truncated:
            result_text += "\n\n" + content[max_chars:]
        
        return result_text.strip()
        
    except Exception as e:
        print(f"   ⚠️ LLM expansion failed: {e}")
        return content  # Return original on failure


def enrich_document(
    input_path: Path,
    output_path: Path,
    client_info: Optional[tuple[str, Any]],
    force: bool = False,
) -> tuple[bool, str]:
    """
    Enrich a single Markdown document.
    
    Args:
        input_path: Source file path
        output_path: Destination file path
        client_info: LLM client info tuple (None for copy-only)
        force: If True, re-enrich even if already processed
    
    Returns:
        (success, status) tuple
    """
    content = input_path.read_text(encoding="utf-8")
    
    # Parse front matter
    metadata = {}
    body = content
    
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
            except yaml.YAMLError:
                pass
    
    # Check if already enriched
    if metadata.get("enriched") and not force:
        if output_path.exists():
            return (True, "skipped (already enriched)")
    
    # Enrich or copy
    if client_info:
        # LLM expansion
        expanded_body = expand_content_with_llm(body, metadata, client_info)
        
        # Update metadata
        metadata["enriched"] = True
        metadata["enriched_at"] = datetime.now().isoformat()
        metadata["enriched_model"] = os.getenv("ENRICHMENT_MODEL", "gpt-4.1")
        
        status = "enriched"
    else:
        # Simple copy
        expanded_body = body
        status = "copied"
    
    # Write output
    yaml_header = yaml.dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    new_content = f"---\n{yaml_header}---\n\n{expanded_body}"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_content, encoding="utf-8")
    
    return (True, status)


def process_all_documents(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> tuple[int, int, int]:
    """
    Process all Markdown documents.
    
    Args:
        input_dir: Directory containing prepared Markdown files
        output_dir: Directory for enriched output
        force: If True, re-process already enriched documents
    
    Returns:
        Tuple of (enriched_count, copied_count, total_count)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Check for LLM client
    client_info = get_enrichment_client()
    
    if client_info:
        model = os.getenv("ENRICHMENT_MODEL", "gpt-4.1")
        print(f"🤖 LLM Content Expansion: {model}")
    else:
        print("📋 No LLM configured - copying files without enrichment")
        print("")
        print("To enable enrichment, set these environment variables:")
        print("  ENRICHMENT_ENDPOINT=https://your-endpoint.inference.ai.azure.com")
        print("  ENRICHMENT_API_KEY=your-api-key")
        print("  ENRICHMENT_MODEL=gpt-4.1  (optional)")
    
    print("=" * 50)
    
    if not input_dir.exists():
        print(f"⚠️ Input directory not found: {input_dir}")
        return (0, 0, 0)
    
    # Find all Markdown files
    md_files = sorted(input_dir.glob("*.md"))
    
    if not md_files:
        print(f"⚠️ No Markdown files found in: {input_dir}")
        return (0, 0, 0)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📚 Found {len(md_files)} documents")
    print(f"📁 Output: {output_dir}")
    
    enriched_count = 0
    copied_count = 0
    
    for i, input_path in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] {input_path.name}")
        
        try:
            output_path = output_dir / input_path.name
            success, status = enrich_document(
                input_path, output_path, client_info, force
            )
            
            if success:
                print(f"   ✅ {status}")
                if status == "enriched":
                    enriched_count += 1
                elif status == "copied":
                    copied_count += 1
                # skipped doesn't increment either counter
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    if client_info:
        print(f"✅ Enriched: {enriched_count}/{len(md_files)} documents")
    else:
        print(f"📋 Copied: {copied_count}/{len(md_files)} documents")
    print(f"📁 Output: {output_dir}")
    
    return (enriched_count, copied_count, len(md_files))


def main() -> int:
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enrich Markdown documents by expanding terse content into full sentences."
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
        "--force",
        action="store_true",
        help="Re-enrich already processed documents",
    )
    
    args = parser.parse_args()
    
    enriched, copied, total = process_all_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        exit(130)
