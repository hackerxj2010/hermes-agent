"""Shell adapter abstraction for cross-platform command execution.

Provides ``BashShellAdapter`` and ``PowerShellAdapter`` that encapsulate
all shell-specific logic: command wrapping, environment snapshots, CWD
tracking, process lifecycle, and path conventions.

Usage::

    adapter = get_shell_adapter()          # auto-detect
    adapter = get_shell_adapter("bash")    # explicit
    adapter = get_shell_adapter("powershell")
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import shutil
import signal
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ShellAdapter(ABC):
    """Abstract interface for shell-specific operations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable shell name (e.g. 'bash', 'powershell')."""
        ...

    @property
    @abstractmethod
    def platform_label(self) -> str:
        """Label for the LLM tool description (e.g. 'Linux', 'Windows')."""
        ...

    @abstractmethod
    def find_shell(self) -> str:
        """Return the absolute path to the shell binary."""
        ...

    @abstractmethod
    def build_run_args(self, cmd_string: str, *, login: bool = False) -> list[str]:
        """Build the full argv list for subprocess.Popen."""
        ...

    @abstractmethod
    def build_snapshot_script(self, snapshot_path: str, cwd_file: str, cwd_marker: str) -> str:
        """Return a shell script that captures the session snapshot."""
        ...

    @abstractmethod
    def wrap_command(
        self,
        command: str,
        cwd: str,
        snapshot_path: str,
        cwd_file: str,
        cwd_marker: str,
        snapshot_ready: bool,
    ) -> str:
        """Wrap a user command with snapshot sourcing, CWD tracking, etc."""
        ...

    @abstractmethod
    def prepend_init_files(self, cmd_string: str, files: list[str]) -> str:
        """Prepend shell init file sourcing to a command string."""
        ...

    @abstractmethod
    def get_temp_dir(self) -> str:
        """Return a writable temp directory path for this platform."""
        ...

    @abstractmethod
    def get_sane_path(self) -> str:
        """Return a sensible default PATH for environments with minimal PATH."""
        ...

    @abstractmethod
    def path_separator(self) -> str:
        """Return the PATH separator character (':' or ';')."""
        ...

    @abstractmethod
    def make_run_env(self, env: dict, blocklist: frozenset, passthrough_fn) -> dict:
        """Build the subprocess environment with PATH sanity and variable filtering."""
        ...

    @abstractmethod
    def kill_process(self, proc: subprocess.Popen) -> None:
        """Kill a process and its children."""
        ...

    @abstractmethod
    def get_tool_description(self) -> str:
        """Return the terminal tool description for the LLM."""
        ...

    def get_preexec_fn(self):
        """Return preexec_fn for Popen (setsid on Linux, None on Windows)."""
        return None


# ---------------------------------------------------------------------------
# Bash adapter (Linux / macOS / Git Bash on Windows)
# ---------------------------------------------------------------------------


class BashShellAdapter(ShellAdapter):
    """Shell adapter for bash (Linux, macOS, Git Bash on Windows)."""

    @property
    def name(self) -> str:
        return "bash"

    @property
    def platform_label(self) -> str:
        return "Linux"

    def find_shell(self) -> str:
        if not _IS_WINDOWS:
            return (
                shutil.which("bash")
                or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
                or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
                or os.environ.get("SHELL")
                or "/bin/sh"
            )

        # Windows: Git Bash
        custom = os.environ.get("HERMES_GIT_BASH_PATH")
        if custom and os.path.isfile(custom):
            return custom

        found = shutil.which("bash")
        if found:
            return found

        for candidate in (
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
        ):
            if candidate and os.path.isfile(candidate):
                return candidate

        raise RuntimeError(
            "Git Bash not found. Hermes Agent requires Git for Windows when using bash on Windows.\n"
            "Install it from: https://git-scm.com/download/win\n"
            "Or set HERMES_GIT_BASH_PATH to your bash.exe location.\n"
            "Alternatively, run 'hermes setup' and select PowerShell as your shell."
        )

    def build_run_args(self, cmd_string: str, *, login: bool = False) -> list[str]:
        shell = self.find_shell()
        if login:
            return [shell, "-l", "-c", cmd_string]
        return [shell, "-c", cmd_string]

    def build_snapshot_script(self, snapshot_path: str, cwd_file: str, cwd_marker: str) -> str:
        return (
            f"export -p > {snapshot_path}\n"
            f"declare -f | grep -vE '^_[^_]' >> {snapshot_path}\n"
            f"alias -p >> {snapshot_path}\n"
            f"echo 'shopt -s expand_aliases' >> {snapshot_path}\n"
            f"echo 'set +e' >> {snapshot_path}\n"
            f"echo 'set +u' >> {snapshot_path}\n"
            f"pwd -P > {cwd_file} 2>/dev/null || true\n"
            f"printf '\\n{cwd_marker}%s{cwd_marker}\\n' \"$(pwd -P)\"\n"
        )

    def wrap_command(
        self,
        command: str,
        cwd: str,
        snapshot_path: str,
        cwd_file: str,
        cwd_marker: str,
        snapshot_ready: bool,
    ) -> str:
        escaped = command.replace("'", "'\\''")
        parts = []

        if snapshot_ready:
            parts.append(f"source {snapshot_path} 2>/dev/null || true")

        quoted_cwd = (
            shlex.quote(cwd) if cwd != "~" and not cwd.startswith("~/") else cwd
        )
        parts.append(f"cd {quoted_cwd} || exit 126")
        parts.append(f"eval '{escaped}'")
        parts.append("__hermes_ec=$?")

        if snapshot_ready:
            parts.append(f"export -p > {snapshot_path} 2>/dev/null || true")

        parts.append(f"pwd -P > {cwd_file} 2>/dev/null || true")
        parts.append(
            f"printf '\\n{cwd_marker}%s{cwd_marker}\\n' \"$(pwd -P)\""
        )
        parts.append("exit $__hermes_ec")

        return "\n".join(parts)

    def prepend_init_files(self, cmd_string: str, files: list[str]) -> str:
        if not files:
            return cmd_string
        prelude_parts = ["set +e"]
        for path in files:
            safe = path.replace("'", "'\\''")
            prelude_parts.append(f"[ -r '{safe}' ] && . '{safe}' 2>/dev/null || true")
        prelude = "\n".join(prelude_parts) + "\n"
        return prelude + cmd_string

    def get_temp_dir(self) -> str:
        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return candidate.rstrip("/") or "/"
        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"
        import tempfile
        candidate = tempfile.gettempdir()
        if candidate.startswith("/"):
            return candidate.rstrip("/") or "/"
        return "/tmp"

    def get_sane_path(self) -> str:
        return (
            "/opt/homebrew/bin:/opt/homebrew/sbin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )

    def path_separator(self) -> str:
        return ":"

    def make_run_env(self, env: dict, blocklist: frozenset, passthrough_fn) -> dict:
        from hermes_constants import get_subprocess_home
        _HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"

        merged = dict(os.environ | env)
        run_env = {}
        for k, v in merged.items():
            if k.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
                real_key = k[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
                run_env[real_key] = v
            elif k not in blocklist or passthrough_fn(k):
                run_env[k] = v

        existing_path = run_env.get("PATH", "")
        sane = self.get_sane_path()
        if "/usr/bin" not in existing_path.split(":"):
            run_env["PATH"] = f"{existing_path}:{sane}" if existing_path else sane

        _profile_home = get_subprocess_home()
        if _profile_home:
            run_env["HOME"] = _profile_home

        return run_env

    def kill_process(self, proc: subprocess.Popen) -> None:
        try:
            if _IS_WINDOWS:
                proc.terminate()
            else:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except Exception:
                pass

    def get_preexec_fn(self):
        if _IS_WINDOWS:
            return None
        return os.setsid

    def get_tool_description(self) -> str:
        return (
            "Execute shell commands on a Linux environment. "
            "Filesystem usually persists between calls.\n\n"
            "Do NOT use cat/head/tail to read files — use read_file instead.\n"
            "Do NOT use grep/rg/find to search — use search_files instead.\n"
            "Do NOT use ls to list directories — use search_files(target='files') instead.\n"
            "Do NOT use sed/awk to edit files — use patch instead.\n"
            "Do NOT use echo/cat heredoc to create files — use write_file instead.\n"
            "Reserve terminal for: builds, installs, git, processes, scripts, network, "
            "package managers, and anything that needs a shell.\n\n"
            "Foreground (default): Commands return INSTANTLY when done, even if the timeout is high. "
            "Set timeout=300 for long builds/scripts — you'll still get the result in seconds if it's fast. "
            "Prefer foreground for short commands.\n"
            "Background: Set background=true to get a session_id. Two patterns:\n"
            "  (1) Long-lived processes that never exit (servers, watchers).\n"
            "  (2) Long-running tasks with notify_on_complete=true — you can keep working on other things "
            "and the system auto-notifies you when the task finishes. Great for test suites, builds, deployments, "
            "or anything that takes more than a minute.\n"
            "For servers/watchers, do NOT use shell-level background wrappers (nohup/disown/setsid/trailing '&') "
            "in foreground mode. Use background=true so Hermes can track lifecycle and output.\n"
            "After starting a server, verify readiness with a health check or log signal, then run tests in a "
            "separate terminal() call. Avoid blind sleep loops.\n"
            "Use process(action=\"poll\") for progress checks, process(action=\"wait\") to block until done.\n"
            "Working directory: Use 'workdir' for per-command cwd.\n"
            "PTY mode: Set pty=true for interactive CLI tools (Codex, Claude Code, Python REPL).\n\n"
            "Do NOT use vim/nano/interactive tools without pty=true — they hang without a pseudo-terminal. "
            "Pipe git output to cat if it might page."
        )


# ---------------------------------------------------------------------------
# PowerShell adapter (native Windows)
# ---------------------------------------------------------------------------


class PowerShellAdapter(ShellAdapter):
    """Shell adapter for PowerShell (native Windows execution)."""

    @property
    def name(self) -> str:
        return "powershell"

    @property
    def platform_label(self) -> str:
        return "Windows"

    def find_shell(self) -> str:
        # Prefer pwsh (PowerShell 7+) over powershell.exe (5.1)
        custom = os.environ.get("HERMES_POWERSHELL_PATH")
        if custom and os.path.isfile(custom):
            return custom

        pwsh = shutil.which("pwsh")
        if pwsh:
            return pwsh

        ps = shutil.which("powershell")
        if ps:
            return ps

        # Fallback to known paths
        for candidate in (
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "PowerShell", "7", "pwsh.exe"),
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
        ):
            if candidate and os.path.isfile(candidate):
                return candidate

        raise RuntimeError(
            "PowerShell not found. Hermes Agent requires PowerShell on Windows.\n"
            "Install PowerShell 7: https://aka.ms/powershell\n"
            "Or set HERMES_POWERSHELL_PATH to your pwsh.exe location.\n"
            "Alternatively, run 'hermes setup' and select Bash (Git Bash) as your shell."
        )

    def build_run_args(self, cmd_string: str, *, login: bool = False) -> list[str]:
        shell = self.find_shell()
        # -NoProfile speeds up startup; -NonInteractive prevents prompts
        # -ExecutionPolicy Bypass ensures scripts run without policy blocks
        return [
            shell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy", "Bypass",
            "-Command", cmd_string,
        ]

    def build_snapshot_script(self, snapshot_path: str, cwd_file: str, cwd_marker: str) -> str:
        # Capture environment variables as a PowerShell script that re-sets them
        # Use a simpler approach: dump env vars as $env:NAME = 'VALUE' lines
        return (
            f"$ErrorActionPreference = 'SilentlyContinue'\n"
            f"Get-ChildItem Env: | ForEach-Object {{ "
            f"'$env:' + $_.Name + ' = ' + [char]39 + ($_.Value -replace [char]39, ([char]39+[char]39)) + [char]39 "
            f"}} | Out-File -FilePath '{snapshot_path}' -Encoding UTF8\n"
            f"(Get-Location).Path | Out-File -FilePath '{cwd_file}' -Encoding UTF8 -NoNewline\n"
            f"Write-Output \"`n{cwd_marker}$((Get-Location).Path){cwd_marker}\"\n"
        )

    def wrap_command(
        self,
        command: str,
        cwd: str,
        snapshot_path: str,
        cwd_file: str,
        cwd_marker: str,
        snapshot_ready: bool,
    ) -> str:
        parts = []

        parts.append("$ErrorActionPreference = 'Continue'")

        # Source snapshot (re-apply env vars from previous commands)
        if snapshot_ready:
            parts.append(
                f"if (Test-Path '{snapshot_path}') {{ . '{snapshot_path}' }}"
            )

        # Change to working directory
        # Handle ~ expansion for PowerShell
        if cwd == "~" or cwd.startswith("~/"):
            ps_cwd = cwd.replace("~", "$HOME", 1)
        else:
            ps_cwd = cwd
        parts.append(f"Set-Location -Path '{ps_cwd}' -ErrorAction Stop")

        # Execute the user command using Invoke-Expression for complex commands
        # Simpler commands can run directly, but eval equivalent is safer
        parts.append(f"Invoke-Expression {_ps_escape_string(command)}")
        parts.append("$__hermes_ec = $LASTEXITCODE; if ($null -eq $__hermes_ec) { $__hermes_ec = if ($?) {{ 0 }} else {{ 1 }} }")

        # Re-dump env vars to snapshot
        if snapshot_ready:
            parts.append(
                f"Get-ChildItem Env: | ForEach-Object {{ "
                f"'$env:' + $_.Name + ' = ' + [char]39 + ($_.Value -replace [char]39, ([char]39+[char]39)) + [char]39 "
                f"}} | Out-File -FilePath '{snapshot_path}' -Encoding UTF8"
            )

        # Write CWD
        parts.append(
            f"(Get-Location).Path | Out-File -FilePath '{cwd_file}' -Encoding UTF8 -NoNewline"
        )
        parts.append(
            f"Write-Output \"`n{cwd_marker}$((Get-Location).Path){cwd_marker}\""
        )
        parts.append("exit $__hermes_ec")

        return "\n".join(parts)

    def prepend_init_files(self, cmd_string: str, files: list[str]) -> str:
        if not files:
            return cmd_string
        prelude_parts = ["$ErrorActionPreference = 'SilentlyContinue'"]
        for path in files:
            safe = path.replace("'", "''")
            prelude_parts.append(f"if (Test-Path '{safe}') {{ . '{safe}' }}")
        prelude_parts.append("$ErrorActionPreference = 'Continue'")
        prelude = "\n".join(prelude_parts) + "\n"
        return prelude + cmd_string

    def get_temp_dir(self) -> str:
        # On Windows, use the system temp directory
        for env_var in ("TEMP", "TMP"):
            candidate = os.environ.get(env_var)
            if candidate and os.path.isdir(candidate):
                return candidate
        import tempfile
        return tempfile.gettempdir()

    def get_sane_path(self) -> str:
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        return os.pathsep.join([
            os.path.join(system_root, "System32"),
            system_root,
            os.path.join(system_root, "System32", "Wbem"),
            os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "PowerShell", "7"),
        ])

    def path_separator(self) -> str:
        return ";"

    def make_run_env(self, env: dict, blocklist: frozenset, passthrough_fn) -> dict:
        from hermes_constants import get_subprocess_home
        _HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"

        merged = dict(os.environ | env)
        run_env = {}
        for k, v in merged.items():
            if k.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
                real_key = k[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
                run_env[real_key] = v
            elif k not in blocklist or passthrough_fn(k):
                run_env[k] = v

        # Ensure System32 is on PATH
        existing_path = run_env.get("PATH", run_env.get("Path", ""))
        system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        if system32.lower() not in existing_path.lower():
            sane = self.get_sane_path()
            run_env["PATH"] = f"{existing_path};{sane}" if existing_path else sane

        _profile_home = get_subprocess_home()
        if _profile_home:
            run_env["USERPROFILE"] = _profile_home
            run_env["HOME"] = _profile_home

        return run_env

    def kill_process(self, proc: subprocess.Popen) -> None:
        try:
            # On Windows, use taskkill to kill the process tree
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def get_tool_description(self) -> str:
        return (
            "Execute PowerShell commands on a Windows environment. "
            "Use PowerShell syntax and cmdlets. Filesystem persists between calls.\n\n"
            "IMPORTANT: This is a Windows/PowerShell environment. Use PowerShell commands:\n"
            "  - Use Get-ChildItem (dir/ls), Set-Location (cd), Get-Content, etc.\n"
            "  - Use $env:VAR to access environment variables\n"
            "  - Use ; to chain commands (not &&)\n"
            "  - Use Select-String instead of grep\n"
            "  - Use Invoke-WebRequest instead of curl/wget\n"
            "  - Paths use backslash (\\) as separator\n\n"
            "Do NOT use cat/head/tail to read files — use read_file instead.\n"
            "Do NOT use grep/rg/find to search — use search_files instead.\n"
            "Do NOT use ls to list directories — use search_files(target='files') instead.\n"
            "Do NOT use sed/awk to edit files — use patch instead.\n"
            "Do NOT use echo/cat heredoc to create files — use write_file instead.\n"
            "Reserve terminal for: builds, installs, git, processes, scripts, network, "
            "package managers, and anything that needs a shell.\n\n"
            "Foreground (default): Commands return INSTANTLY when done, even if the timeout is high. "
            "Set timeout=300 for long builds/scripts — you'll still get the result in seconds if it's fast. "
            "Prefer foreground for short commands.\n"
            "Background: Set background=true to get a session_id. Two patterns:\n"
            "  (1) Long-lived processes that never exit (servers, watchers).\n"
            "  (2) Long-running tasks with notify_on_complete=true — the system auto-notifies "
            "you when the task finishes.\n"
            "Use process(action=\"poll\") for progress checks, process(action=\"wait\") to block until done.\n"
            "Working directory: Use 'workdir' for per-command cwd.\n"
            "PTY mode: Set pty=true for interactive CLI tools."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ps_escape_string(s: str) -> str:
    """Escape a string for safe use inside PowerShell Invoke-Expression.

    Uses a here-string-like approach with double quotes and backtick escaping.
    """
    # For Invoke-Expression, wrap in double quotes and escape internal doubles
    escaped = s.replace("'", "''")
    return f"'{escaped}'"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Cache the adapter instance for the process lifetime
_cached_adapter: ShellAdapter | None = None


def get_shell_adapter(shell_type: str | None = None) -> ShellAdapter:
    """Return the appropriate shell adapter.

    Args:
        shell_type: One of 'bash', 'powershell', or None for auto-detect.
            Auto-detect reads TERMINAL_SHELL env var, then falls back to
            'powershell' on Windows and 'bash' on everything else.

    Returns:
        A ShellAdapter instance.
    """
    global _cached_adapter

    if shell_type is None:
        shell_type = os.environ.get("TERMINAL_SHELL", "").lower()

    if not shell_type:
        # Auto-detect: PowerShell on Windows, bash everywhere else
        if _IS_WINDOWS:
            shell_type = "powershell"
        else:
            shell_type = "bash"

    if shell_type in ("powershell", "pwsh"):
        if _cached_adapter is None or _cached_adapter.name != "powershell":
            _cached_adapter = PowerShellAdapter()
        return _cached_adapter
    else:
        if _cached_adapter is None or _cached_adapter.name != "bash":
            _cached_adapter = BashShellAdapter()
        return _cached_adapter


def get_shell_for_env_type(env_type: str) -> ShellAdapter:
    """Return the correct shell adapter for a given environment type.

    Container/remote backends (docker, modal, ssh, daytona, singularity)
    always use bash because they run inside Linux. Only 'local' respects
    the user's shell preference.
    """
    if env_type == "local":
        return get_shell_adapter()  # user preference
    # All container/remote backends are Linux → always bash
    return BashShellAdapter()
