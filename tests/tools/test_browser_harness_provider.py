from unittest.mock import Mock, patch

import pytest

from tools.browser_providers.browser_harness import BrowserHarnessProvider


def test_is_configured_true_when_explicit_cdp_url_present(monkeypatch):
    provider = BrowserHarnessProvider()
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    monkeypatch.setattr(provider, "_discover_local_cdp_url", lambda: (None, []))

    assert provider.is_configured() is True


def test_is_configured_false_when_no_endpoint_found(monkeypatch):
    provider = BrowserHarnessProvider()
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(provider, "_discover_local_cdp_url", lambda: (None, ["C:/missing/DevToolsActivePort"]))

    assert provider.is_configured() is False
    assert provider._last_checked_paths == ["C:/missing/DevToolsActivePort"]


def test_resolve_cdp_url_auto_detects_devtools_active_port(tmp_path, monkeypatch):
    root = tmp_path / "Chrome" / "User Data" / "Default"
    root.mkdir(parents=True)
    (root / "DevToolsActivePort").write_text("9222\n/devtools/browser/abc123\n", encoding="utf-8")

    provider = BrowserHarnessProvider()
    monkeypatch.setattr(provider, "_candidate_profile_roots", lambda: [tmp_path / "Chrome" / "User Data"])
    monkeypatch.setattr(provider, "_is_loopback_port_open", lambda port: port == 9222)

    assert provider.resolve_cdp_url() == "ws://127.0.0.1:9222/devtools/browser/abc123"


def test_resolve_cdp_url_normalizes_explicit_http_endpoint(monkeypatch):
    provider = BrowserHarnessProvider()
    monkeypatch.setenv("BROWSER_CDP_URL", "http://localhost:9222")
    monkeypatch.setattr(provider, "_discover_local_cdp_url", lambda: (None, []))

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "webSocketDebuggerUrl": "ws://localhost:9222/devtools/browser/live",
    }

    with patch("tools.browser_providers.browser_harness.requests.get", return_value=response) as mock_get:
        resolved = provider.resolve_cdp_url()

    assert resolved == "ws://localhost:9222/devtools/browser/live"
    mock_get.assert_called_once_with("http://localhost:9222/json/version", timeout=3)


def test_create_session_error_mentions_checked_paths(monkeypatch):
    provider = BrowserHarnessProvider()
    provider._last_checked_paths = [
        "C:/Users/test/AppData/Local/Google/Chrome/User Data/Default/DevToolsActivePort",
        "C:/Users/test/AppData/Local/Microsoft/Edge/User Data/Default/DevToolsActivePort",
    ]
    monkeypatch.setattr(provider, "_resolve_cdp_url_with_source", lambda: ("", ""))

    with pytest.raises(RuntimeError, match="Checked: C:/Users/test/AppData/Local/Google/Chrome/User Data/Default/DevToolsActivePort"):
        provider.create_session("task-1")
