"""Tests for the hybrid local-only web backend configuration."""

import json
from unittest.mock import patch


class TestBackendSelection:
    def test_backend_is_always_searxng(self):
        from tools.web_tools import _get_backend

        assert _get_backend() == "searxng"

    def test_backend_ignores_legacy_config_values(self):
        from tools.web_tools import _get_backend

        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}):
            assert _get_backend() == "searxng"


class TestWebRequirements:
    def test_web_requires_only_searxng_url(self):
        from tools.web_tools import _web_requires_env

        assert _web_requires_env() == ["SEARXNG_URL"]

    def test_check_web_api_key_requires_searxng_url(self):
        from tools.web_tools import check_web_api_key

        with patch.dict("os.environ", {"SEARXNG_URL": "http://localhost:8080"}, clear=False), \
             patch("tools.web_tools._searxng_is_reachable", return_value=True):
            assert check_web_api_key() is True

        with patch.dict("os.environ", {"SEARXNG_URL": ""}, clear=False):
            assert check_web_api_key() is False

    def test_check_web_api_key_false_when_searxng_unreachable(self):
        from tools.web_tools import check_web_api_key

        with patch.dict("os.environ", {"SEARXNG_URL": "http://localhost:8080"}, clear=False), \
             patch("tools.web_tools._searxng_is_reachable", return_value=False):
            assert check_web_api_key() is False


class TestWebSearch:
    def test_web_search_uses_searxng_backend(self):
        import tools.web_tools as web_tools

        fake_results = {
            "query": "hermes",
            "data": {"web": [{"title": "Hermes", "url": "https://example.com"}]},
        }

        with patch("tools.web_tools._searxng_search", return_value=fake_results), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(web_tools._debug, "log_call"), \
             patch.object(web_tools._debug, "save"):
            result = json.loads(web_tools.web_search_tool("hermes", limit=3))

        assert result["data"]["web"][0]["url"] == "https://example.com"

    def test_web_search_error_response_is_minimal(self):
        import tools.web_tools as web_tools

        with patch("tools.web_tools._searxng_search", side_effect=RuntimeError("boom")), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(web_tools._debug, "log_call") as mock_log_call, \
             patch.object(web_tools._debug, "save"):
            result = json.loads(web_tools.web_search_tool("hermes", limit=3))

        assert result == {"error": "Error searching web: boom"}
        debug_payload = mock_log_call.call_args.args[1]
        assert debug_payload["error"] == "Error searching web: boom"
