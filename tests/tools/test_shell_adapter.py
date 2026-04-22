"""Tests for the shell adapter abstraction layer."""

import os
import platform
import subprocess
from unittest import mock

import pytest

from tools.environments.shell_adapter import (
    BashShellAdapter,
    PowerShellAdapter,
    ShellAdapter,
    get_shell_adapter,
    get_shell_for_env_type,
)


_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# BashShellAdapter
# ---------------------------------------------------------------------------


class TestBashShellAdapter:
    def setup_method(self):
        self.adapter = BashShellAdapter()

    def test_name(self):
        assert self.adapter.name == "bash"

    def test_platform_label(self):
        assert self.adapter.platform_label == "Linux"

    def test_path_separator(self):
        assert self.adapter.path_separator() == ":"

    def test_get_sane_path_contains_usr_bin(self):
        sane = self.adapter.get_sane_path()
        assert "/usr/bin" in sane

    def test_build_run_args_basic(self):
        with mock.patch.object(self.adapter, "find_shell", return_value="/bin/bash"):
            args = self.adapter.build_run_args("echo hello")
            assert args == ["/bin/bash", "-c", "echo hello"]

    def test_build_run_args_login(self):
        with mock.patch.object(self.adapter, "find_shell", return_value="/bin/bash"):
            args = self.adapter.build_run_args("echo hello", login=True)
            assert args == ["/bin/bash", "-l", "-c", "echo hello"]

    def test_build_snapshot_script_contains_export(self):
        script = self.adapter.build_snapshot_script("/tmp/snap.sh", "/tmp/cwd.txt", "__CWD__")
        assert "export -p" in script
        assert "__CWD__" in script
        assert "/tmp/snap.sh" in script

    def test_wrap_command_contains_cd(self):
        wrapped = self.adapter.wrap_command(
            command="ls -la",
            cwd="/home/user",
            snapshot_path="/tmp/snap.sh",
            cwd_file="/tmp/cwd.txt",
            cwd_marker="__CWD__",
            snapshot_ready=True,
        )
        assert "cd" in wrapped
        assert "/home/user" in wrapped
        assert "source /tmp/snap.sh" in wrapped
        assert "ls -la" in wrapped

    def test_wrap_command_no_snapshot(self):
        wrapped = self.adapter.wrap_command(
            command="pwd",
            cwd="/tmp",
            snapshot_path="/tmp/snap.sh",
            cwd_file="/tmp/cwd.txt",
            cwd_marker="__CWD__",
            snapshot_ready=False,
        )
        assert "source" not in wrapped
        assert "pwd" in wrapped

    def test_prepend_init_files_empty(self):
        result = self.adapter.prepend_init_files("echo hi", [])
        assert result == "echo hi"

    def test_prepend_init_files_with_files(self):
        result = self.adapter.prepend_init_files("echo hi", ["/home/user/.bashrc"])
        assert ".bashrc" in result
        assert "echo hi" in result

    def test_get_temp_dir_returns_string(self):
        temp = self.adapter.get_temp_dir()
        assert isinstance(temp, str)
        assert len(temp) > 0

    def test_tool_description_mentions_linux(self):
        desc = self.adapter.get_tool_description()
        assert "Linux" in desc

    @pytest.mark.skipif(_IS_WINDOWS, reason="setsid not available on Windows")
    def test_preexec_fn_is_setsid_on_linux(self):
        fn = self.adapter.get_preexec_fn()
        assert fn is os.setsid

    @pytest.mark.skipif(not _IS_WINDOWS, reason="Only on Windows")
    def test_preexec_fn_is_none_on_windows(self):
        fn = self.adapter.get_preexec_fn()
        assert fn is None


# ---------------------------------------------------------------------------
# PowerShellAdapter
# ---------------------------------------------------------------------------


class TestPowerShellAdapter:
    def setup_method(self):
        self.adapter = PowerShellAdapter()

    def test_name(self):
        assert self.adapter.name == "powershell"

    def test_platform_label(self):
        assert self.adapter.platform_label == "Windows"

    def test_path_separator(self):
        assert self.adapter.path_separator() == ";"

    def test_build_run_args_contains_noprofile(self):
        with mock.patch.object(self.adapter, "find_shell", return_value="pwsh.exe"):
            args = self.adapter.build_run_args("Get-Date")
            assert "pwsh.exe" in args
            assert "-NoProfile" in args
            assert "-Command" in args
            assert "Get-Date" in args

    def test_build_snapshot_script_contains_env(self):
        script = self.adapter.build_snapshot_script(
            "C:\\temp\\snap.ps1", "C:\\temp\\cwd.txt", "__CWD__"
        )
        assert "Get-ChildItem Env:" in script
        assert "__CWD__" in script

    def test_wrap_command_contains_set_location(self):
        wrapped = self.adapter.wrap_command(
            command="Get-ChildItem",
            cwd="C:\\Users\\Test",
            snapshot_path="C:\\temp\\snap.ps1",
            cwd_file="C:\\temp\\cwd.txt",
            cwd_marker="__CWD__",
            snapshot_ready=True,
        )
        assert "Set-Location" in wrapped
        assert "Get-ChildItem" in wrapped

    def test_wrap_command_handles_tilde(self):
        wrapped = self.adapter.wrap_command(
            command="dir",
            cwd="~",
            snapshot_path="C:\\temp\\snap.ps1",
            cwd_file="C:\\temp\\cwd.txt",
            cwd_marker="__CWD__",
            snapshot_ready=False,
        )
        assert "$HOME" in wrapped

    def test_prepend_init_files_empty(self):
        result = self.adapter.prepend_init_files("Get-Date", [])
        assert result == "Get-Date"

    def test_prepend_init_files_with_files(self):
        result = self.adapter.prepend_init_files("Get-Date", ["C:\\profile.ps1"])
        assert "profile.ps1" in result
        assert "Get-Date" in result

    def test_get_temp_dir_returns_string(self):
        temp = self.adapter.get_temp_dir()
        assert isinstance(temp, str)
        assert len(temp) > 0

    def test_get_sane_path_contains_system32(self):
        sane = self.adapter.get_sane_path()
        assert "System32" in sane

    def test_tool_description_mentions_powershell(self):
        desc = self.adapter.get_tool_description()
        assert "PowerShell" in desc
        assert "Windows" in desc

    def test_preexec_fn_is_none(self):
        fn = self.adapter.get_preexec_fn()
        assert fn is None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestGetShellAdapter:
    def test_explicit_bash(self):
        adapter = get_shell_adapter("bash")
        assert adapter.name == "bash"

    def test_explicit_powershell(self):
        adapter = get_shell_adapter("powershell")
        assert adapter.name == "powershell"

    def test_explicit_pwsh(self):
        adapter = get_shell_adapter("pwsh")
        assert adapter.name == "powershell"

    def test_env_var_override(self):
        with mock.patch.dict(os.environ, {"TERMINAL_SHELL": "powershell"}):
            # Clear cache
            import tools.environments.shell_adapter as sa
            sa._cached_adapter = None
            adapter = get_shell_adapter()
            assert adapter.name == "powershell"
            sa._cached_adapter = None

    def test_env_var_bash(self):
        with mock.patch.dict(os.environ, {"TERMINAL_SHELL": "bash"}):
            import tools.environments.shell_adapter as sa
            sa._cached_adapter = None
            adapter = get_shell_adapter()
            assert adapter.name == "bash"
            sa._cached_adapter = None


class TestGetShellForEnvType:
    def test_local_respects_preference(self):
        with mock.patch.dict(os.environ, {"TERMINAL_SHELL": "powershell"}):
            import tools.environments.shell_adapter as sa
            sa._cached_adapter = None
            adapter = get_shell_for_env_type("local")
            assert adapter.name == "powershell"
            sa._cached_adapter = None

    def test_docker_always_bash(self):
        adapter = get_shell_for_env_type("docker")
        assert adapter.name == "bash"

    def test_ssh_always_bash(self):
        adapter = get_shell_for_env_type("ssh")
        assert adapter.name == "bash"

    def test_modal_always_bash(self):
        adapter = get_shell_for_env_type("modal")
        assert adapter.name == "bash"

    def test_singularity_always_bash(self):
        adapter = get_shell_for_env_type("singularity")
        assert adapter.name == "bash"

    def test_daytona_always_bash(self):
        adapter = get_shell_for_env_type("daytona")
        assert adapter.name == "bash"
