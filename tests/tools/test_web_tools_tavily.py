"""Runtime tests for the local-only SearxNG web backend."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import tools.web_tools as web_tools


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str):
        request = httpx.Request("GET", url)
        if url == "https://example.com/docs":
            return httpx.Response(
                200,
                text="<html><title>Docs</title><body>Hello from docs</body></html>",
                request=request,
            )
        return httpx.Response(404, text="missing", request=request)


def test_searxng_search_filters_duplicates_and_disallowed_urls():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "results": [
            {"url": "https://example.com/a", "title": "A", "content": "desc-a"},
            {"url": "https://example.com/a#fragment", "title": "A dup", "content": "dup"},
            {"url": "http://127.0.0.1:8080/internal", "title": "Internal", "content": "bad"},
            {"url": "https://blocked.example/page", "title": "Blocked", "content": "blocked"},
            {"url": "https://example.com/b", "title": "B", "content": "desc-b"},
        ]
    }

    with patch("tools.web_tools.httpx.get", return_value=response) as mock_get, \
         patch("tools.web_tools.is_safe_url", side_effect=lambda url: "127.0.0.1" not in url), \
         patch(
             "tools.web_tools.check_website_access",
             side_effect=lambda url: {"message": "blocked"} if "blocked.example" in url else None,
         ):
        result = web_tools._searxng_search("hermes", limit=10)

    assert result["success"] is True
    assert result["data"]["web"] == [
        {
            "url": "https://example.com/a",
            "title": "A",
            "description": "desc-a",
            "position": 1,
        },
        {
            "url": "https://example.com/b",
            "title": "B",
            "description": "desc-b",
            "position": 2,
        },
    ]
    assert mock_get.call_args.kwargs["headers"] == web_tools._WEB_REQUEST_HEADERS


@pytest.mark.asyncio
async def test_web_extract_blocks_policy_and_fetches_pages():
    with patch("tools.web_tools.httpx.AsyncClient", _FakeAsyncClient), \
         patch("tools.web_tools.check_auxiliary_model", return_value=False), \
         patch(
             "tools.web_tools.check_website_access",
             side_effect=lambda url: {"message": "Blocked by website policy"} if "blocked.example" in url else None,
         ), \
         patch("tools.web_tools.is_safe_url", return_value=True), \
         patch.object(web_tools._debug, "log_call"), \
         patch.object(web_tools._debug, "save"):
        raw = await web_tools.web_extract_tool(
            ["https://blocked.example/page", "https://example.com/docs"],
            use_llm_processing=False,
        )

    result = json.loads(raw)
    assert result["results"][0]["error"] == "Blocked by website policy"
    assert result["results"][0]["blocked_by_policy"]["message"] == "Blocked by website policy"
    assert result["results"][1]["title"] == "Docs"
    assert "Hello from docs" in result["results"][1]["content"]


@pytest.mark.asyncio
async def test_web_crawl_uses_site_scoped_query_and_filters_offsite_results():
    extract_mock = AsyncMock(return_value=json.dumps({"results": [{"url": "https://example.com"}]}))
    search_results = {
        "success": True,
        "data": {
            "web": [
                {"url": "https://docs.example.com/start", "title": "Docs", "description": "docs", "position": 1},
                {"url": "https://other.example.net/offsite", "title": "Offsite", "description": "offsite", "position": 2},
                {"url": "https://example.com/blog", "title": "Blog", "description": "blog", "position": 3},
            ]
        },
    }

    with patch("tools.web_tools._searxng_search", return_value=search_results) as mock_search, \
         patch("tools.web_tools.web_extract_tool", extract_mock), \
         patch("tools.web_tools.is_safe_url", return_value=True), \
         patch("tools.web_tools.check_website_access", return_value=None):
        result = await web_tools.web_crawl_tool(
            "example.com",
            instructions="release notes",
            depth="basic",
            use_llm_processing=False,
        )

    assert json.loads(result)["results"][0]["url"] == "https://example.com"
    mock_search.assert_called_once_with("site:example.com release notes", limit=10)
    assert extract_mock.await_args.args[0] == [
        "https://example.com",
        "https://docs.example.com/start",
        "https://example.com/blog",
    ]


@pytest.mark.asyncio
async def test_web_crawl_advanced_depth_increases_page_budget():
    extract_mock = AsyncMock(return_value=json.dumps({"results": []}))
    search_results = {
        "success": True,
        "data": {
            "web": [
                {"url": f"https://example.com/page-{idx}", "title": f"Page {idx}", "description": "x", "position": idx}
                for idx in range(1, 15)
            ]
        },
    }

    with patch("tools.web_tools._searxng_search", return_value=search_results) as mock_search, \
         patch("tools.web_tools.web_extract_tool", extract_mock), \
         patch("tools.web_tools.is_safe_url", return_value=True), \
         patch("tools.web_tools.check_website_access", return_value=None):
        await web_tools.web_crawl_tool(
            "https://example.com",
            depth="advanced",
            use_llm_processing=False,
        )

    mock_search.assert_called_once_with("site:example.com", limit=20)
    assert len(extract_mock.await_args.args[0]) == 10
