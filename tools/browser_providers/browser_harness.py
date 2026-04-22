"""Browser Harness provider for local browser automation."""

from __future__ import annotations

import logging
import os
import re
import socket
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests

from tools.browser_providers.base import CloudBrowserProvider

logger = logging.getLogger(__name__)

_DISCOVERY_TIMEOUT_SECONDS = 3
_SOCKET_TIMEOUT_SECONDS = 0.5


class BrowserHarnessProvider(CloudBrowserProvider):
    """Provider for local browser automation using Browser Harness / CDP."""

    def __init__(self) -> None:
        self._last_checked_paths: list[str] = []

    def provider_name(self) -> str:
        return "browser-harness"

    def is_configured(self) -> bool:
        """Return True when a live local/browser-attached CDP endpoint is discoverable."""
        if self._explicit_cdp_url():
            return True
        cdp_url, checked_paths = self._discover_local_cdp_url()
        self._last_checked_paths = checked_paths
        return bool(cdp_url)

    def _explicit_cdp_url(self) -> str:
        return (os.getenv("BROWSER_CDP_URL") or "").strip()

    def _candidate_profile_roots(self) -> Iterable[Path]:
        home = Path.home()
        local_appdata = Path(os.getenv("LOCALAPPDATA", str(home / "AppData" / "Local")))

        roots = [
            # Windows
            local_appdata / "Google" / "Chrome" / "User Data",
            local_appdata / "Google" / "Chrome Beta" / "User Data",
            local_appdata / "Google" / "Chrome Dev" / "User Data",
            local_appdata / "Google" / "Chrome SxS" / "User Data",
            local_appdata / "Chromium" / "User Data",
            local_appdata / "Microsoft" / "Edge" / "User Data",
            local_appdata / "Microsoft" / "Edge Beta" / "User Data",
            local_appdata / "Microsoft" / "Edge Dev" / "User Data",
            local_appdata / "Microsoft" / "Edge SxS" / "User Data",
            # macOS
            home / "Library" / "Application Support" / "Google" / "Chrome",
            home / "Library" / "Application Support" / "Google" / "Chrome Beta",
            home / "Library" / "Application Support" / "Google" / "Chrome Dev",
            home / "Library" / "Application Support" / "Google" / "Chrome Canary",
            home / "Library" / "Application Support" / "Chromium",
            home / "Library" / "Application Support" / "Microsoft Edge",
            home / "Library" / "Application Support" / "Microsoft Edge Beta",
            home / "Library" / "Application Support" / "Microsoft Edge Dev",
            home / "Library" / "Application Support" / "Microsoft Edge Canary",
            # Linux / BSD
            home / ".config" / "google-chrome",
            home / ".config" / "google-chrome-beta",
            home / ".config" / "google-chrome-unstable",
            home / ".config" / "chromium",
            home / ".var" / "app" / "com.google.Chrome" / "config" / "google-chrome",
            home / ".var" / "app" / "org.chromium.Chromium" / "config" / "chromium",
            Path("/tmp"),
            Path("/private/tmp"),
        ]

        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            yield root

    def _iter_devtools_activeport_files(self) -> Iterable[Path]:
        seen: set[str] = set()

        def _emit(path: Path) -> Iterable[Path]:
            key = str(path)
            if key in seen:
                return []
            seen.add(key)
            return [path]

        for root in self._candidate_profile_roots():
            for candidate in _emit(root / "DevToolsActivePort"):
                yield candidate

            if not root.exists() or not root.is_dir():
                continue

            for name in ("Default", "Guest Profile", "System Profile"):
                for candidate in _emit(root / name / "DevToolsActivePort"):
                    yield candidate

            try:
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if child.name.startswith("Profile ") or child.name.endswith(" Profile"):
                        for candidate in _emit(child / "DevToolsActivePort"):
                            yield candidate
            except OSError:
                continue

    def _read_devtools_activeport(self, candidate: Path) -> Optional[tuple[int, str]]:
        try:
            raw = candidate.read_text(encoding="utf-8").strip().splitlines()
        except (OSError, UnicodeDecodeError):
            return None

        if len(raw) < 2:
            return None

        try:
            port = int(raw[0].strip())
        except ValueError:
            return None

        path = raw[1].strip()
        if not path.startswith("/"):
            path = f"/{path.lstrip('/')}"
        return port, path

    def _is_loopback_port_open(self, port: int) -> bool:
        for host in ("127.0.0.1", "localhost"):
            try:
                with socket.create_connection((host, port), timeout=_SOCKET_TIMEOUT_SECONDS):
                    return True
            except OSError:
                continue
        return False

    def _discover_local_cdp_url(self) -> tuple[Optional[str], list[str]]:
        checked_paths: list[str] = []

        for candidate in self._iter_devtools_activeport_files():
            checked_paths.append(str(candidate))
            parsed = self._read_devtools_activeport(candidate)
            if parsed is None:
                continue
            port, path = parsed
            if self._is_loopback_port_open(port):
                cdp_url = f"ws://127.0.0.1:{port}{path}"
                logger.info(
                    "BrowserHarness auto-detected CDP endpoint: %s (from %s)",
                    cdp_url, candidate,
                )
                return cdp_url, checked_paths

        if checked_paths:
            logger.debug(
                "BrowserHarness: no live CDP endpoint found (checked %d paths)",
                len(checked_paths),
            )
        return None, checked_paths

    def _normalize_cdp_url(self, cdp_url: str) -> str:
        raw = (cdp_url or "").strip()
        if not raw:
            return ""

        lowered = raw.lower()
        if "/devtools/browser/" in lowered:
            return raw

        discovery_url = raw
        if lowered.startswith(("ws://", "wss://")):
            ws_host = raw.split("://", 1)[1].strip("/")
            if "/" not in ws_host and ":" in ws_host:
                discovery_url = ("http://" if lowered.startswith("ws://") else "https://") + ws_host
            else:
                return raw
        elif "://" not in raw and re.match(r"^[^/]+:\d+$", raw):
            discovery_url = f"http://{raw}"

        version_url = (
            discovery_url
            if discovery_url.lower().endswith("/json/version")
            else discovery_url.rstrip("/") + "/json/version"
        )

        try:
            response = requests.get(version_url, timeout=_DISCOVERY_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("BrowserHarness discovery probe failed for %s: %s", version_url, exc)
            return raw

        websocket_url = str(payload.get("webSocketDebuggerUrl", "") or "").strip()
        return websocket_url or raw

    def resolve_cdp_url(self) -> str:
        local_cdp_url, checked_paths = self._discover_local_cdp_url()
        self._last_checked_paths = checked_paths
        if local_cdp_url:
            return local_cdp_url

        explicit_cdp_url = self._explicit_cdp_url()
        if explicit_cdp_url:
            return self._normalize_cdp_url(explicit_cdp_url)

        return ""

    def _resolve_cdp_url_with_source(self) -> tuple[str, str]:
        local_cdp_url, checked_paths = self._discover_local_cdp_url()
        self._last_checked_paths = checked_paths
        if local_cdp_url:
            return local_cdp_url, "autodetect"

        explicit_cdp_url = self._explicit_cdp_url()
        if explicit_cdp_url:
            return self._normalize_cdp_url(explicit_cdp_url), "explicit"

        return "", ""

    def create_session(self, task_id: str) -> Dict[str, object]:
        """Connect to a local browser via CDP."""
        cdp_url, cdp_source = self._resolve_cdp_url_with_source()
        if not cdp_url:
            checked_paths = ", ".join(self._last_checked_paths[:6])
            if len(self._last_checked_paths) > 6:
                checked_paths += ", ..."
            checked_suffix = f" Checked: {checked_paths}." if checked_paths else ""
            raise RuntimeError(
                "Browser-Harness could not find a running browser with remote debugging enabled. "
                "Start Chrome/Edge/Chromium with --remote-debugging-port=9222, "
                "use '/browser connect', or set BROWSER_CDP_URL explicitly."
                + checked_suffix
            )

        return {
            "session_name": f"harness-{task_id}",
            "bb_session_id": f"local-{task_id}",
            "cdp_url": cdp_url,
            "features": {
                "is_local": True,
                "browser_harness": True,
                "cdp_source": cdp_source,
            },
        }

    def close_session(self, session_id: str) -> bool:
        """Local sessions don't need explicit closing through a provider."""
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        """Nothing to clean up for local CDP sessions."""
        return None

    def diagnostic_info(self) -> Dict[str, object]:
        """Return a summary of the provider state for diagnostics / logging."""
        cdp_url, source = self._resolve_cdp_url_with_source()
        return {
            "provider": self.provider_name(),
            "cdp_url": cdp_url or None,
            "cdp_source": source or None,
            "explicit_env": bool(self._explicit_cdp_url()),
            "checked_paths_count": len(self._last_checked_paths),
        }
