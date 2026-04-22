#!/usr/bin/env python3
"""
Standalone Web Tools Module

This module provides generic web tools that work with local SearxNG backend.
Backend is selected during ``hermes tools`` setup (web.backend in config.yaml).

Available tools:
- web_search_tool: Search the web for information
- web_extract_tool: Extract content from specific web pages
- web_crawl_tool: Crawl websites with specific instructions (via Browser-Harness)

LLM Processing:
- Uses frontier models (Gemini 3, OpenRouter) or local models for intelligent extraction
- Extracts key excerpts and creates markdown summaries to reduce token usage

Debug Mode:
- Set WEB_TOOLS_DEBUG=true to enable detailed logging
- Creates web_tools_debug_UUID.json in ./logs directory
- Captures all tool calls, results, and compression metrics

Usage:
    from web_tools import web_search_tool, web_extract_tool, web_crawl_tool
    
    # Search the web
    results = web_search_tool("Python machine learning libraries", limit=3)
    
    # Extract content from URLs  
    content = web_extract_tool(["https://example.com"], format="markdown")
"""

import json
import logging
import os
import re
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import httpx
from agent.auxiliary_client import (
    async_call_llm,
    extract_content_or_reasoning,
    get_async_text_auxiliary_client,
)
from tools.debug_helpers import DebugSession
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


# ─── Backend Selection ────────────────────────────────────────────────────────

def _has_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip())

def _load_web_config() -> dict:
    """Load the ``web:`` section from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web", {})
    except (ImportError, Exception):
        return {}

def _get_backend() -> str:
    """Determine which web backend to use (always searxng)."""
    return "searxng"

def _is_backend_available(backend: str) -> bool:
    """Return True when the selected backend is currently usable."""
    if backend == "searxng":
        return _has_env("SEARXNG_URL")
    return False

def _web_requires_env() -> list[str]:
    """Return tool metadata env vars for SearxNG."""
    return ["SEARXNG_URL"]


_SEARXNG_SEARCH_TIMEOUT_SECONDS = 12
_SEARXNG_RETRY_DELAY_SECONDS = 1.5
_WEB_FETCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_WEB_REQUEST_HEADERS = {
    "User-Agent": "hermes-agent/1.0 (+https://github.com/NousResearch/hermes-agent)",
    "Accept": "application/json,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}


def _get_searxng_base_url() -> str:
    return ((os.getenv("SEARXNG_URL") or "").strip() or "http://localhost:8080").rstrip("/")


def _searxng_is_reachable() -> bool:
    """Quick socket-level probe to check if the SearxNG host is reachable.

    Does NOT issue a full HTTP request — only verifies that the TCP port is
    open.  This keeps the check fast enough for tool-registration time.
    """
    import socket
    url = _get_searxng_base_url()
    try:
        from urllib.parse import urlparse as _urlparse
        parsed = _urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        logger.debug("SearxNG health-check failed for %s", url)
        return False


def _normalize_result_limit(limit: int, *, default: int = 5, maximum: int = 10) -> int:
    try:
        numeric = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(numeric, maximum))


def _canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"


def _is_allowed_result_url(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    if not is_safe_url(url):
        return False
    blocked = check_website_access(url)
    if blocked:
        logger.debug("Skipping policy-blocked search result %s (%s)", url, blocked.get("rule"))
        return False
    return True


def _extract_site_host(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.hostname or "").lower().rstrip(".")


def _is_same_site_url(candidate_url: str, root_host: str) -> bool:
    if not root_host:
        return False
    candidate_host = (urlparse(candidate_url).hostname or "").lower().rstrip(".")
    if not candidate_host:
        return False
    return candidate_host == root_host or candidate_host.endswith(f".{root_host}")


# Extraction helpers


DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000

def _is_nous_auxiliary_client(client: Any) -> bool:
    """Return True when the resolved auxiliary backend is Nous Portal."""
    from urllib.parse import urlparse

    base_url = str(getattr(client, "base_url", "") or "")
    host = (urlparse(base_url).hostname or "").lower()
    return host == "nousresearch.com" or host.endswith(".nousresearch.com")


def _resolve_web_extract_auxiliary(model: Optional[str] = None) -> tuple[Optional[Any], Optional[str], Dict[str, Any]]:
    """Resolve the current web-extract auxiliary client, model, and extra body."""
    client, default_model = get_async_text_auxiliary_client("web_extract")
    configured_model = os.getenv("AUXILIARY_WEB_EXTRACT_MODEL", "").strip()
    effective_model = model or configured_model or default_model

    extra_body: Dict[str, Any] = {}
    if client is not None and _is_nous_auxiliary_client(client):
        from agent.auxiliary_client import get_auxiliary_extra_body
        extra_body = get_auxiliary_extra_body() or {"tags": ["product=hermes-agent"]}

    return client, effective_model, extra_body


def _get_default_summarizer_model() -> Optional[str]:
    """Return the current default model for web extraction summarization."""
    _, model, _ = _resolve_web_extract_auxiliary()
    return model

_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


async def process_content_with_llm(
    content: str, 
    url: str = "", 
    title: str = "",
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> Optional[str]:
    """
    Process web content using LLM to create intelligent summaries with key excerpts.
    
    This function uses frontier models (Gemini 3, OpenRouter) or local models
    to intelligently extract key information and create markdown summaries,
    significantly reducing token usage while preserving all important information.
    
    For very large content (>500k chars), uses chunked processing with synthesis.
    For extremely large content (>2M chars), refuses to process entirely.
    
    Args:
        content (str): The raw content to process
        url (str): The source URL (for context, optional)
        title (str): The page title (for context, optional)
        model (str): The model to use for processing
        min_length (int): Minimum content length to trigger processing (default: 5000)
        
    Returns:
        Optional[str]: Processed markdown content, or None if content too short or processing fails
    """
    # Size thresholds
    MAX_CONTENT_SIZE = 2_000_000  # 2M chars - refuse entirely above this
    CHUNK_THRESHOLD = 500_000     # 500k chars - use chunked processing above this
    CHUNK_SIZE = 100_000          # 100k chars per chunk
    MAX_OUTPUT_SIZE = 5000        # Hard cap on final output size
    
    try:
        content_len = len(content)
        
        # Refuse if content is absurdly large
        if content_len > MAX_CONTENT_SIZE:
            size_mb = content_len / 1_000_000
            logger.warning("Content too large (%.1fMB > 2MB limit). Refusing to process.", size_mb)
            return f"[Content too large to process: {size_mb:.1f}MB. Try using web_crawl with specific extraction instructions, or search for a more focused source.]"
        
        # Skip processing if content is too short
        if content_len < min_length:
            logger.debug("Content too short (%d < %d chars), skipping LLM processing", content_len, min_length)
            return None
        
        # Create context information
        context_info = []
        if title:
            context_info.append(f"Title: {title}")
        if url:
            context_info.append(f"Source: {url}")
        context_str = "\n".join(context_info) + "\n\n" if context_info else ""
        
        # Check if we need chunked processing
        if content_len > CHUNK_THRESHOLD:
            logger.info("Content large (%d chars). Using chunked processing...", content_len)
            return await _process_large_content_chunked(
                content, context_str, model, CHUNK_SIZE, MAX_OUTPUT_SIZE
            )
        
        # Standard single-pass processing for normal content
        logger.info("Processing content with LLM (%d characters)", content_len)
        
        processed_content = await _call_summarizer_llm(content, context_str, model)
        
        if processed_content:
            # Enforce output cap
            if len(processed_content) > MAX_OUTPUT_SIZE:
                processed_content = processed_content[:MAX_OUTPUT_SIZE] + "\n\n[... summary truncated for context management ...]"
            
            # Log compression metrics
            processed_length = len(processed_content)
            compression_ratio = processed_length / content_len if content_len > 0 else 1.0
            logger.info("Content processed: %d -> %d chars (%.1f%%)", content_len, processed_length, compression_ratio * 100)
        
        return processed_content
        
    except Exception as e:
        logger.warning(
            "web_extract LLM summarization failed (%s). "
            "Tip: increase auxiliary.web_extract.timeout in config.yaml "
            "or switch to a faster auxiliary model.",
            str(e)[:120],
        )
        # Fall back to truncated raw content instead of returning a useless
        # error message.  The first ~5000 chars are almost always more useful
        # to the model than "[Failed to process content: ...]".
        truncated = content[:MAX_OUTPUT_SIZE]
        if len(content) > MAX_OUTPUT_SIZE:
            truncated += (
                f"\n\n[Content truncated — showing first {MAX_OUTPUT_SIZE:,} of "
                f"{len(content):,} chars. LLM summarization timed out. "
                f"To fix: increase auxiliary.web_extract.timeout in config.yaml, "
                f"or use a faster auxiliary model. Use browser_navigate for the full page.]"
            )
        return truncated


async def _call_summarizer_llm(
    content: str, 
    context_str: str, 
    model: Optional[str], 
    max_tokens: int = 20000,
    is_chunk: bool = False,
    chunk_info: str = ""
) -> Optional[str]:
    """
    Make a single LLM call to summarize content.
    
    Args:
        content: The content to summarize
        context_str: Context information (title, URL)
        model: Model to use
        max_tokens: Maximum output tokens
        is_chunk: Whether this is a chunk of a larger document
        chunk_info: Information about chunk position (e.g., "Chunk 2/5")
        
    Returns:
        Summarized content or None on failure
    """
    if is_chunk:
        # Chunk-specific prompt - aware that this is partial content
        system_prompt = """You are an expert content analyst processing a SECTION of a larger document. Your job is to extract and summarize the key information from THIS SECTION ONLY.

Important guidelines for chunk processing:
1. Do NOT write introductions or conclusions - this is a partial document
2. Focus on extracting ALL key facts, figures, data points, and insights from this section
3. Preserve important quotes, code snippets, and specific details verbatim
4. Use bullet points and structured formatting for easy synthesis later
5. Note any references to other sections (e.g., "as mentioned earlier", "see below") without trying to resolve them

Your output will be combined with summaries of other sections, so focus on thorough extraction rather than narrative flow."""

        user_prompt = f"""Extract key information from this SECTION of a larger document:

{context_str}{chunk_info}

SECTION CONTENT:
{content}

Extract all important information from this section in a structured format. Focus on facts, data, insights, and key details. Do not add introductions or conclusions."""

    else:
        # Standard full-document prompt
        system_prompt = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

        user_prompt = f"""Please process this web content and create a comprehensive markdown summary:

{context_str}CONTENT TO PROCESS:
{content}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""

    # Call the LLM with retry logic — keep retries low since summarization
    # is a nice-to-have; the caller falls back to truncated content on failure.
    max_retries = 2
    retry_delay = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
            if aux_client is None or not effective_model:
                logger.warning("No auxiliary model available for web content processing")
                return None
            call_kwargs = {
                "task": "web_extract",
                "model": effective_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                # No explicit timeout — async_call_llm reads auxiliary.web_extract.timeout
                # from config (default 360s / 6min).  Users with slow local models can
                # increase it in config.yaml.
            }
            if extra_body:
                call_kwargs["extra_body"] = extra_body
            response = await async_call_llm(**call_kwargs)
            content = extract_content_or_reasoning(response)
            if content:
                return content
            # Reasoning-only / empty response — let the retry loop handle it
            logger.warning("LLM returned empty content (attempt %d/%d), retrying", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue
            return content  # Return whatever we got after exhausting retries
        except RuntimeError:
            logger.warning("No auxiliary model available for web content processing")
            return None
        except Exception as api_error:
            last_error = api_error
            if attempt < max_retries - 1:
                logger.warning("LLM API call failed (attempt %d/%d): %s", attempt + 1, max_retries, str(api_error)[:100])
                logger.warning("Retrying in %ds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                raise last_error
    
    return None


async def _process_large_content_chunked(
    content: str, 
    context_str: str, 
    model: Optional[str], 
    chunk_size: int,
    max_output_size: int
) -> Optional[str]:
    """
    Process large content by chunking, summarizing each chunk in parallel,
    then synthesizing the summaries.
    
    Args:
        content: The large content to process
        context_str: Context information
        model: Model to use
        chunk_size: Size of each chunk in characters
        max_output_size: Maximum final output size
        
    Returns:
        Synthesized summary or None on failure
    """
    # Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info("Split into %d chunks of ~%d chars each", len(chunks), chunk_size)
    
    # Summarize each chunk in parallel
    async def summarize_chunk(chunk_idx: int, chunk_content: str) -> tuple[int, Optional[str]]:
        """Summarize a single chunk."""
        try:
            chunk_info = f"[Processing chunk {chunk_idx + 1} of {len(chunks)}]"
            summary = await _call_summarizer_llm(
                chunk_content, 
                context_str, 
                model, 
                max_tokens=10000,
                is_chunk=True,
                chunk_info=chunk_info
            )
            if summary:
                logger.info("Chunk %d/%d summarized: %d -> %d chars", chunk_idx + 1, len(chunks), len(chunk_content), len(summary))
            return chunk_idx, summary
        except Exception as e:
            logger.warning("Chunk %d/%d failed: %s", chunk_idx + 1, len(chunks), str(e)[:50])
            return chunk_idx, None
    
    # Run all chunk summarizations in parallel
    tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    
    # Collect successful summaries in order
    summaries = []
    for chunk_idx, summary in sorted(results, key=lambda x: x[0]):
        if summary:
            summaries.append(f"## Section {chunk_idx + 1}\n{summary}")
    
    if not summaries:
        logger.debug("All chunk summarizations failed")
        return "[Failed to process large content: all chunk summarizations failed]"
    
    logger.info("Got %d/%d chunk summaries", len(summaries), len(chunks))
    
    # If only one chunk succeeded, just return it (with cap)
    if len(summaries) == 1:
        result = summaries[0]
        if len(result) > max_output_size:
            result = result[:max_output_size] + "\n\n[... truncated ...]"
        return result
    
    # Synthesize the summaries into a final summary
    logger.info("Synthesizing %d summaries...", len(summaries))
    
    combined_summaries = "\n\n---\n\n".join(summaries)
    
    synthesis_prompt = f"""You have been given summaries of different sections of a large document. 
Synthesize these into ONE cohesive, comprehensive summary that:
1. Removes redundancy between sections
2. Preserves all key facts, figures, and actionable information
3. Is well-organized with clear structure
4. Is under {max_output_size} characters

{context_str}SECTION SUMMARIES:
{combined_summaries}

Create a single, unified markdown summary."""

    try:
        aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
        if aux_client is None or not effective_model:
            logger.warning("No auxiliary model for synthesis, concatenating summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        call_kwargs = {
            "task": "web_extract",
            "model": effective_model,
            "messages": [
                {"role": "system", "content": "You synthesize multiple summaries into one cohesive, comprehensive summary. Be thorough but concise."},
                {"role": "user", "content": synthesis_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 20000,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        response = await async_call_llm(**call_kwargs)
        final_summary = extract_content_or_reasoning(response)

        # Retry once on empty content (reasoning-only response)
        if not final_summary:
            logger.warning("Synthesis LLM returned empty content, retrying once")
            response = await async_call_llm(**call_kwargs)
            final_summary = extract_content_or_reasoning(response)

        # If still None after retry, fall back to concatenated summaries
        if not final_summary:
            logger.warning("Synthesis failed after retry — concatenating chunk summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        # Enforce hard cap
        if len(final_summary) > max_output_size:
            final_summary = final_summary[:max_output_size] + "\n\n[... summary truncated for context management ...]"
        
        original_len = len(content)
        final_len = len(final_summary)
        compression = final_len / original_len if original_len > 0 else 1.0
        
        logger.info("Synthesis complete: %d -> %d chars (%.2f%%)", original_len, final_len, compression * 100)
        return final_summary
        
    except Exception as e:
        logger.warning("Synthesis failed: %s", str(e)[:100])
        # Fall back to concatenated summaries with truncation
        fallback = "\n\n".join(summaries)
        if len(fallback) > max_output_size:
            fallback = fallback[:max_output_size] + "\n\n[... truncated due to synthesis failure ...]"
        return fallback


def clean_base64_images(text: str) -> str:
    """
    Remove base64 encoded images from text to reduce token count and clutter.
    
    This function finds and removes base64 encoded images in various formats:
    - (data:image/png;base64,...)
    - (data:image/jpeg;base64,...)
    - (data:image/svg+xml;base64,...)
    - data:image/[type];base64,... (without parentheses)
    
    Args:
        text: The text content to clean
        
    Returns:
        Cleaned text with base64 images replaced with placeholders
    """
    # Pattern to match base64 encoded images wrapped in parentheses
    # Matches: (data:image/[type];base64,[base64-string])
    base64_with_parens_pattern = r'\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'
    
    # Pattern to match base64 encoded images without parentheses
    # Matches: data:image/[type];base64,[base64-string]
    base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    
    # Replace parentheses-wrapped images first
    cleaned_text = re.sub(base64_with_parens_pattern, '[BASE64_IMAGE_REMOVED]', text)
    
    # Then replace any remaining non-parentheses images
    cleaned_text = re.sub(base64_pattern, '[BASE64_IMAGE_REMOVED]', cleaned_text)
    
    return cleaned_text


# ─── SearxNG Client ──────────────────────────────────────────────────────────

def _searxng_search(query: str, limit: int = 10) -> dict:
    """Search using local SearxNG and return results as a dict.

    Includes one retry with backoff for transient failures (e.g. SearxNG
    container cold-starting).
    """
    searxng_url = _get_searxng_base_url()
    safe_limit = _normalize_result_limit(limit, default=5, maximum=10)
    last_error: Exception | None = None

    for attempt in range(2):  # 1 initial + 1 retry
        try:
            if attempt > 0:
                import time
                time.sleep(_SEARXNG_RETRY_DELAY_SECONDS)
                logger.info("SearxNG search retry %d for '%s'", attempt, query)

            logger.info("SearxNG search: '%s' (limit=%d)", query, safe_limit)
            response = httpx.get(
                f"{searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "pageno": 1,
                },
                headers=_WEB_REQUEST_HEADERS,
                timeout=_SEARXNG_SEARCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

            web_results = []
            seen_urls: set[str] = set()
            for result in data.get("results", []):
                url = str(result.get("url", "") or "").strip()
                canonical_url = _canonicalize_url(url)
                if not url or canonical_url in seen_urls or not _is_allowed_result_url(url):
                    continue

                seen_urls.add(canonical_url)
                web_results.append({
                    "url": url,
                    "title": str(result.get("title", "") or ""),
                    "description": str(result.get("content", "") or ""),
                    "position": len(web_results) + 1,
                })
                if len(web_results) >= safe_limit:
                    break

            return {"success": True, "data": {"web": web_results}}

        except Exception as e:
            last_error = e
            if attempt == 0:
                logger.warning("SearxNG search attempt failed, will retry: %s", e)
            continue

    logger.error("SearxNG search failed after retry: %s", last_error)
    return {"success": False, "error": str(last_error), "data": {"web": []}}





def web_search_tool(query: str, limit: int = 5) -> str:
    """
    Search the web for information using available search API backend.

    This function provides a generic interface for web search using local SearxNG.

    Note: This function returns search result metadata only (URLs, titles, descriptions).
    Use web_extract_tool to get full content from specific URLs.
    
    Args:
        query (str): The search query to look up
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        str: JSON string containing search results with the following structure:
             {
                 "success": bool,
                 "data": {
                     "web": [
                         {
                             "title": str,
                             "url": str,
                             "description": str,
                             "position": int
                         },
                         ...
                     ]
                 }
             }
    
    Raises:
        Exception: If search fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "query": query,
            "limit": limit
        },
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0
    }
    
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        # Always use SearxNG
        response_data = _searxng_search(query, limit)
        debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        debug_call_data["final_response_size"] = len(result_json)
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()
        return result_json
        
    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        logger.debug("%s", error_msg)

        debug_call_data["error"] = error_msg
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()

        return tool_error(error_msg)


async def web_extract_tool(
    urls: List[str],
    format: str = None,
    use_llm_processing: bool = True,
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> str:
    """
    Extract content from specific web pages using available extraction API backend.

    This function provides a generic interface for web content extraction that
    can work with multiple backends. Currently uses SearxNG.

    Args:
        urls (List[str]): List of URLs to extract content from
        format (str): Desired output format ("markdown" or "html", optional)
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (Optional[str]): The model to use for LLM processing (defaults to current auxiliary backend model)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)

    Security: URLs are checked for embedded secrets before fetching.
    
    Returns:
        str: JSON string containing extracted content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
    
    Raises:
        Exception: If extraction fails or API key is not set
    """
    # Block URLs containing embedded secrets (exfiltration prevention).
    # URL-decode first so percent-encoded secrets (%73k- = sk-) are caught.
    from agent.redact import _PREFIX_RE
    from urllib.parse import unquote
    for _url in urls:
        if _PREFIX_RE.search(_url) or _PREFIX_RE.search(unquote(_url)):
            return json.dumps({
                "success": False,
                "error": "Blocked: URL contains what appears to be an API key or token. "
                         "Secrets must not be sent in URLs.",
            })

    debug_call_data = {
        "parameters": {
            "urls": urls,
            "format": format,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length
        },
        "error": None,
        "pages_extracted": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        logger.info("Extracting content from %d URL(s)", len(urls))

        # ── SSRF protection — filter out private/internal URLs before any backend ──
        safe_urls = []
        blocked_results: List[Dict[str, Any]] = []
        for url in urls:
            blocked_by_policy = check_website_access(url)
            if blocked_by_policy:
                blocked_results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": blocked_by_policy.get("message", "Blocked by website policy"),
                    "blocked_by_policy": blocked_by_policy,
                })
            elif not is_safe_url(url):
                blocked_results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                })
            else:
                safe_urls.append(url)

        async def fetch_single_url(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
            try:
                logger.info("Fetching: %s", url)
                response = await client.get(url)
                response.raise_for_status()
                
                final_url = str(response.url)
                blocked_by_policy = check_website_access(final_url)
                if blocked_by_policy:
                    return {
                        "url": final_url,
                        "title": "",
                        "content": "",
                        "error": blocked_by_policy.get("message", "Blocked by website policy"),
                        "blocked_by_policy": blocked_by_policy,
                    }

                content = response.text
                title = ""
                title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()

                return {
                    "url": str(response.url),
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": {"sourceURL": str(response.url), "title": title},
                }
            except Exception as e:
                logger.warning("Fetch failed for %s: %s", url, e)
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": str(e),
                }

        results = list(blocked_results)
        if safe_urls:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=_WEB_FETCH_TIMEOUT,
                headers=_WEB_REQUEST_HEADERS,
            ) as client:
                fetched_results = await asyncio.gather(
                    *(fetch_single_url(client, url) for url in safe_urls)
                )
            results.extend(fetched_results)

        response = {"results": results}
        
        pages_extracted = len(response.get('results', []))
        logger.info("Extracted content from %d pages", pages_extracted)
        
        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = len(json.dumps(response))
        effective_model = model or _get_default_summarizer_model()
        auxiliary_available = check_auxiliary_model()
        
        # Process each result with LLM if enabled
        if use_llm_processing and auxiliary_available:
            logger.info("Processing extracted content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")
            
            # Prepare tasks for parallel processing
            async def process_single_result(result):
                """Process a single result with LLM and return updated result with metrics."""
                url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                raw_content = result.get('raw_content', '') or result.get('content', '')
                
                if not raw_content:
                    return result, None, "no_content"
                
                original_size = len(raw_content)
                
                # Process content with LLM
                processed = await process_content_with_llm(
                    raw_content, url, title, effective_model, min_length
                )
                
                if processed:
                    processed_size = len(processed)
                    compression_ratio = processed_size / original_size if original_size > 0 else 1.0
                    
                    # Update result with processed content
                    result['content'] = processed
                    result['raw_content'] = raw_content
                    
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                        "model_used": effective_model
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "model_used": None,
                        "reason": "content_too_short"
                    }
                    return result, metrics, "too_short"
            
            # Run all LLM processing in parallel
            results_list = response.get('results', [])
            tasks = [process_single_result(result) for result in results_list]
            processed_results = await asyncio.gather(*tasks)
            
            # Collect metrics and print results
            for result, metrics, status in processed_results:
                url = result.get('url', 'Unknown URL')
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", url)
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", url)
                else:
                    logger.warning("%s (no content to process)", url)
        else:
            if use_llm_processing and not auxiliary_available:
                logger.warning("LLM processing requested but no auxiliary model available, returning raw content")
                debug_call_data["processing_applied"].append("llm_processing_unavailable")
            # Print summary of extracted pages for debugging (original behavior)
            for result in response.get('results', []):
                url = result.get('url', 'Unknown URL')
                content_length = len(result.get('raw_content', ''))
                logger.info("%s (%d characters)", url, content_length)
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}

        if trimmed_response.get("results") == []:
            result_json = tool_error("Content was inaccessible or not found")

            cleaned_result = clean_base64_images(result_json)
        
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
            
            cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        return cleaned_result
            
    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        return tool_error(error_msg)


async def web_crawl_tool(
    url: str, 
    instructions: str = None, 
    depth: str = "basic", 
    use_llm_processing: bool = True,
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> str:
    """
    Crawl a website with specific instructions using available crawling API backend.
    
    This function provides a generic interface for web crawling that can work
    can work with multiple backends. Currently uses local browser automation.
    
    Args:
        url (str): The base URL to crawl (can include or exclude https://)
        instructions (str): Instructions for what to crawl/extract using LLM intelligence (optional)
        depth (str): Depth of extraction ("basic" or "advanced", default: "basic")
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (Optional[str]): The model to use for LLM processing (defaults to current auxiliary backend model)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing crawled content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
             Each page is processed individually.
    
    Raises:
        Exception: If crawling fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "url": url,
            "instructions": instructions,
            "depth": depth,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length
        },
        "error": None,
        "pages_crawled": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        effective_model = model or _get_default_summarizer_model()
        site_limit = 10 if str(depth).lower() == "advanced" else 5
        search_limit = 20 if site_limit > 5 else 10

        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'

        logger.info("Local crawl (depth-1) for: %s", url)

        # SSRF protection
        if not is_safe_url(url):
            return json.dumps({"results": [{"url": url, "error": "Blocked: URL targets a private or internal network address"}]}, ensure_ascii=False)
        blocked_by_policy = check_website_access(url)
        if blocked_by_policy:
            return json.dumps({
                "results": [{
                    "url": url,
                    "error": blocked_by_policy.get("message", "Blocked by website policy"),
                    "blocked_by_policy": blocked_by_policy,
                }]
            }, ensure_ascii=False)

        # Local crawl strategy: use site-scoped SearxNG discovery plus extract.
        root_host = _extract_site_host(url)
        query_suffix = (instructions or "").strip()
        search_query = f"site:{root_host} {query_suffix}".strip() if root_host else url
        search_results_data = _searxng_search(search_query, limit=search_limit)

        urls_to_extract = [url]
        seen_urls = {_canonicalize_url(url)}
        for result in search_results_data.get("data", {}).get("web", []):
            candidate_url = str(result.get("url", "") or "").strip()
            canonical_candidate = _canonicalize_url(candidate_url)
            if not candidate_url or canonical_candidate in seen_urls:
                continue
            if root_host and not _is_same_site_url(candidate_url, root_host):
                continue
            seen_urls.add(canonical_candidate)
            urls_to_extract.append(candidate_url)
            if len(urls_to_extract) >= site_limit:
                break

        extracted_content_json = await web_extract_tool(
            urls_to_extract,
            use_llm_processing=use_llm_processing,
            model=effective_model,
            min_length=min_length,
        )
        return extracted_content_json

    except Exception as e:
        error_msg = f"Error during local crawl: {str(e)}"
        logger.error(error_msg)
        return tool_error(error_msg)


def check_web_api_key() -> bool:
    """Check whether SearxNG is available.

    Verifies both that ``SEARXNG_URL`` is set *and* that the host:port is
    TCP-reachable (socket probe, no full HTTP request).  This prevents the
    web tools from being advertised when SearxNG is configured but the
    container/service hasn't started yet.
    """
    if not _has_env("SEARXNG_URL"):
        return False
    return _searxng_is_reachable()


def check_auxiliary_model() -> bool:
    """Check if an auxiliary text model is available for LLM content processing."""
    client, _, _ = _resolve_web_extract_auxiliary()
    return client is not None



if __name__ == "__main__":
    """Simple test/demo when run directly"""
    print("🌐 Standalone Web Tools Module (Local-Only)")
    print("=" * 40)
    
    web_available = check_web_api_key()
    nous_available = check_auxiliary_model()
    default_summarizer_model = _get_default_summarizer_model()

    if web_available:
        print("✅ Web backend: searxng (local)")
    else:
        print("❌ No web search backend configured (SEARXNG_URL not set)")

    if not nous_available:
        print("❌ No auxiliary model available for LLM content processing")
    else:
        print(f"✅ Auxiliary model available: {default_summarizer_model}")

    print("\n🛠️  Web tools ready for local use!")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for information on any topic. Returns up to 5 relevant results with titles, URLs, and descriptions.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web"
            }
        },
        "required": ["query"]
    }
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": "Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs (arxiv papers, documents, etc.) — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized and capped at ~5000 chars per page. Pages over 2M chars are refused. If a URL fails or times out, use the browser tool to access it instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5
            }
        },
        "required": ["urls"]
    }
}

WEB_CRAWL_SCHEMA = {
    "name": "web_crawl",
    "description": "Crawl a website to find relevant information. Returns content from multiple pages on the site. Limited to top 10 pages for local safety.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The base URL to crawl"
            },
            "instructions": {
                "type": "string",
                "description": "Specific instructions for what information to look for"
            },
            "depth": {
                "type": "string",
                "description": "Crawl breadth: 'basic' for up to 5 pages, 'advanced' for up to 10 pages",
                "enum": ["basic", "advanced"],
            }
        },
        "required": ["url"]
    }
}

registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(args.get("query", ""), limit=5),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    emoji="🔍",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_extract",
    toolset="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [], "markdown"),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    is_async=True,
    emoji="📄",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_crawl",
    toolset="web",
    schema=WEB_CRAWL_SCHEMA,
    handler=lambda args, **kw: web_crawl_tool(
        args.get("url", ""),
        args.get("instructions", ""),
        args.get("depth", "basic"),
    ),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    is_async=True,
    emoji="🕷️",
    max_result_size_chars=200_000,
)
