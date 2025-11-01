#!/usr/bin/env python3
"""
Tests for Pre-Commit Hooks Configuration

These tests verify that pre-commit hooks are configured correctly and match CI checks.

Author: Claude Code
Date: 2025-11-01
"""

import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def pre_commit_config(repo_root):
    """Load .pre-commit-config.yaml if it exists."""
    config_path = repo_root / ".pre-commit-config.yaml"
    if not config_path.exists():
        pytest.skip(".pre-commit-config.yaml not yet created")

    with open(config_path) as f:
        return yaml.safe_load(f)


class TestPreCommitConfigExists:
    """Test that pre-commit configuration file exists."""

    def test_pre_commit_config_file_exists(self, repo_root):
        """Test .pre-commit-config.yaml exists."""
        config_path = repo_root / ".pre-commit-config.yaml"
        assert config_path.exists(), ".pre-commit-config.yaml must exist"

    def test_pre_commit_config_valid_yaml(self, repo_root):
        """Test .pre-commit-config.yaml is valid YAML."""
        config_path = repo_root / ".pre-commit-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None, "Config must be valid YAML"
        assert isinstance(config, dict), "Config must be a dictionary"


class TestPreCommitHooksMatchCI:
    """Test that pre-commit hooks match CI checks."""

    def test_ruff_check_hook_present(self, pre_commit_config):
        """Test ruff check hook is configured."""
        repos = pre_commit_config.get("repos", [])
        ruff_repos = [r for r in repos if "ruff" in r.get("repo", "").lower()]

        assert len(ruff_repos) > 0, "Ruff must be in pre-commit hooks"

        # Find ruff-check hook
        ruff_hooks = []
        for repo in ruff_repos:
            ruff_hooks.extend([h for h in repo.get("hooks", []) if "ruff" in h.get("id", "")])

        assert any("check" in h.get("id", "") for h in ruff_hooks), "ruff check hook must exist"

    def test_ruff_format_hook_present(self, pre_commit_config):
        """Test ruff format hook is configured."""
        repos = pre_commit_config.get("repos", [])
        ruff_repos = [r for r in repos if "ruff" in r.get("repo", "").lower()]

        assert len(ruff_repos) > 0, "Ruff must be in pre-commit hooks"

        # Find ruff-format hook
        ruff_hooks = []
        for repo in ruff_repos:
            ruff_hooks.extend([h for h in repo.get("hooks", []) if "ruff" in h.get("id", "")])

        assert any("format" in h.get("id", "") for h in ruff_hooks), "ruff format hook must exist"

    def test_pytest_hook_present(self, pre_commit_config):
        """Test pytest hook is configured."""
        repos = pre_commit_config.get("repos", [])

        # Check for pytest in local hooks or pre-commit hooks
        has_pytest = False
        for repo in repos:
            hooks = repo.get("hooks", [])
            if any("pytest" in h.get("id", "").lower() or "test" in h.get("id", "").lower() for h in hooks):
                has_pytest = True
                break

        assert has_pytest, "pytest or test hook must be configured"

    def test_trailing_whitespace_hook(self, pre_commit_config):
        """Test trailing whitespace hook is configured."""
        repos = pre_commit_config.get("repos", [])

        has_whitespace_hook = False
        for repo in repos:
            hooks = repo.get("hooks", [])
            if any("trailing" in h.get("id", "").lower() and "whitespace" in h.get("id", "").lower() for h in hooks):
                has_whitespace_hook = True
                break

        assert has_whitespace_hook, "Trailing whitespace hook must be configured"

    def test_end_of_file_fixer_hook(self, pre_commit_config):
        """Test end-of-file-fixer hook is configured."""
        repos = pre_commit_config.get("repos", [])

        has_eof_hook = False
        for repo in repos:
            hooks = repo.get("hooks", [])
            if any("end" in h.get("id", "").lower() and "file" in h.get("id", "").lower() for h in hooks):
                has_eof_hook = True
                break

        assert has_eof_hook, "End-of-file-fixer hook must be configured"


class TestPreCommitDependency:
    """Test that pre-commit is in development dependencies."""

    def test_pre_commit_in_requirements_dev(self, repo_root):
        """Test pre-commit is in requirements-dev.txt."""
        req_dev_path = repo_root / "requirements-dev.txt"
        assert req_dev_path.exists(), "requirements-dev.txt must exist"

        with open(req_dev_path) as f:
            content = f.read()

        assert "pre-commit" in content, "pre-commit must be in requirements-dev.txt"


class TestMakefileTargets:
    """Test that Makefile has required targets."""

    def test_makefile_exists(self, repo_root):
        """Test Makefile exists."""
        makefile_path = repo_root / "Makefile"
        assert makefile_path.exists(), "Makefile must exist"

    def test_makefile_ci_target(self, repo_root):
        """Test Makefile has 'ci' target."""
        makefile_path = repo_root / "Makefile"
        with open(makefile_path) as f:
            content = f.read()

        assert "ci:" in content or "ci " in content, "Makefile must have 'ci' target"

    def test_makefile_lint_target(self, repo_root):
        """Test Makefile has 'lint' target."""
        makefile_path = repo_root / "Makefile"
        with open(makefile_path) as f:
            content = f.read()

        assert "lint:" in content or "lint " in content, "Makefile must have 'lint' target"

    def test_makefile_format_target(self, repo_root):
        """Test Makefile has 'format' target."""
        makefile_path = repo_root / "Makefile"
        with open(makefile_path) as f:
            content = f.read()

        assert "format:" in content or "format " in content, "Makefile must have 'format' target"

    def test_makefile_test_target(self, repo_root):
        """Test Makefile has 'test' target."""
        makefile_path = repo_root / "Makefile"
        with open(makefile_path) as f:
            content = f.read()

        assert "test:" in content or "test " in content, "Makefile must have 'test' target"

    def test_makefile_install_hooks_target(self, repo_root):
        """Test Makefile has 'install-hooks' target."""
        makefile_path = repo_root / "Makefile"
        with open(makefile_path) as f:
            content = f.read()

        assert "install-hooks:" in content or "install-hooks " in content, \
            "Makefile must have 'install-hooks' target"


@pytest.mark.integration
class TestPreCommitHooksIntegration:
    """Integration tests for pre-commit hooks (requires pre-commit installed)."""

    def test_pre_commit_run_all_hooks(self, repo_root):
        """Test running all pre-commit hooks."""
        try:
            result = subprocess.run(
                ["pre-commit", "run", "--all-files"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            # Hooks may pass or fail, but command should not error
            assert result.returncode in [0, 1], f"pre-commit run failed: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("pre-commit not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("pre-commit hooks took >60s (requirement: <10s for typical commits)")

    def test_pre_commit_hooks_fast_enough(self, repo_root):
        """Test pre-commit hooks run in <10 seconds on typical file."""
        import time

        try:
            # Test on a single Python file
            test_file = repo_root / "src" / "strategy.py"

            start = time.perf_counter()
            result = subprocess.run(
                ["pre-commit", "run", "--files", str(test_file)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=15
            )
            elapsed = time.perf_counter() - start

            assert elapsed < 10.0, f"Pre-commit hooks took {elapsed:.1f}s (requirement: <10s)"
        except FileNotFoundError:
            pytest.skip("pre-commit not installed")


class TestContributingGuide:
    """Test that CONTRIBUTING.md exists and has required sections."""

    def test_contributing_md_exists(self, repo_root):
        """Test .github/CONTRIBUTING.md exists."""
        contrib_path = repo_root / ".github" / "CONTRIBUTING.md"
        assert contrib_path.exists(), ".github/CONTRIBUTING.md must exist"

    def test_contributing_has_pre_commit_setup(self, repo_root):
        """Test CONTRIBUTING.md mentions pre-commit setup."""
        contrib_path = repo_root / ".github" / "CONTRIBUTING.md"
        with open(contrib_path) as f:
            content = f.read().lower()

        assert "pre-commit" in content, "CONTRIBUTING.md must mention pre-commit"
        assert "install" in content or "setup" in content, \
            "CONTRIBUTING.md must explain how to install hooks"

    def test_contributing_has_local_ci_section(self, repo_root):
        """Test CONTRIBUTING.md has local CI validation section."""
        contrib_path = repo_root / ".github" / "CONTRIBUTING.md"
        with open(contrib_path) as f:
            content = f.read().lower()

        assert "local" in content and ("ci" in content or "validation" in content), \
            "CONTRIBUTING.md must explain local CI validation"
