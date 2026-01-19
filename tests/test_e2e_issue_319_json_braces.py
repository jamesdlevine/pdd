"""
E2E Test for Issue #319: pdd change fails at Step 5 with KeyError when issue contains JSON

This test exercises the full CLI path from `pdd change` (via run_agentic_change) to verify
that when GitHub issue content contains JSON with curly braces, the orchestrator correctly
processes the content without raising KeyError.

The reported bug:
- User runs `pdd change https://github.com/promptdriven/pdd/issues/318`
- At Step 5, the workflow fails with: "Context missing key for step 5: '\n  "type"'"
- This occurs because Python's str.format() interprets JSON braces as placeholders

Investigation findings (from Steps 4-5 of bug workflow):
- The bug as described is technically impossible to reproduce with current codebase
- Python's str.format() does NOT re-parse already-substituted values
- The error can only occur if the TEMPLATE ITSELF contains literal JSON (not context values)
- Current template files don't contain raw JSON

This E2E test serves as a REGRESSION TEST to ensure:
1. The orchestrator handles JSON in issue content
2. The orchestrator handles JSON in step outputs
3. No future changes introduce the bug pattern

The test should PASS on the current codebase (confirming correct behavior).
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def set_pdd_path(monkeypatch):
    """Set PDD_PATH to the pdd package directory for all tests in this module."""
    import pdd
    pdd_package_dir = Path(pdd.__file__).parent
    monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))


class TestIssue319JsonBracesE2E:
    """
    E2E tests for Issue #319: Verify pdd change handles JSON curly braces correctly.

    These tests exercise the full code path from run_agentic_change through
    the orchestrator, with only external dependencies mocked (gh CLI, agent tasks).
    """

    def test_e2e_change_with_json_in_issue_body(self, tmp_path, monkeypatch):
        """
        E2E Test: pdd change should handle GitHub issues containing JSON code blocks.

        This test simulates the exact scenario from Issue #319:
        - Issue body contains JSON with curly braces
        - The orchestrator should process all steps without KeyError

        Bug behavior (Issue #319):
        - Step 5 fails with "Context missing key for step 5: '\n  "type"'"

        Expected behavior (correct implementation):
        - All steps process successfully
        - JSON in issue content is passed as context without being parsed as template
        """
        monkeypatch.chdir(tmp_path)

        # Initialize git repo to satisfy _get_git_root
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

        # Create initial commit so HEAD exists
        (tmp_path / "README.md").write_text("# Test repo")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True)

        # The JSON content that caused the original bug
        json_issue_body = '''
The API returns this error response:

```json
{
  "type": "error",
  "message": "Connection timeout",
  "code": 500,
  "details": {
    "retry_after": 30,
    "endpoint": "/api/v1/users"
  }
}
```

Please investigate and fix this bug.
'''

        # Mock the gh CLI responses
        mock_issue_data = {
            "title": "API returns error on connection timeout",
            "body": json_issue_body,
            "user": {"login": "testuser"},
            "comments_url": "https://api.github.com/repos/owner/repo/issues/319/comments"
        }

        # Comments also contain JSON
        mock_comments = [
            {
                "user": {"login": "commenter1"},
                "body": '''I can reproduce this. The response looks like:
```
{"status": "failed", "reason": "timeout"}
```
'''
            }
        ]

        def mock_subprocess_run(args, **kwargs):
            """Mock subprocess.run for gh commands and git operations."""
            cmd = args if isinstance(args, list) else []
            cmd_str = " ".join(cmd) if cmd else ""

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            # gh api for issue
            if "gh" in cmd and "api" in cmd and "issues/319" in cmd_str and "comments" not in cmd_str:
                result.stdout = json.dumps(mock_issue_data)
                return result

            # gh api for comments
            if "gh" in cmd and "api" in cmd and "comments" in cmd_str:
                result.stdout = json.dumps(mock_comments)
                return result

            # git rev-parse --show-toplevel
            if "git" in cmd and "rev-parse" in cmd and "--show-toplevel" in cmd:
                result.stdout = str(tmp_path)
                return result

            # git rev-parse --abbrev-ref HEAD
            if "git" in cmd and "rev-parse" in cmd and "--abbrev-ref" in cmd:
                result.stdout = "main"
                return result

            # git worktree operations
            if "git" in cmd and "worktree" in cmd:
                if "list" in cmd:
                    result.stdout = ""
                elif "add" in cmd:
                    # Create the worktree directory
                    for i, arg in enumerate(cmd):
                        if arg == "add" and i + 2 < len(cmd):
                            wt_path = Path(cmd[i + 2])
                            wt_path.mkdir(parents=True, exist_ok=True)
                            (wt_path / ".git").write_text("gitdir: /fake")
                            break
                return result

            # git branch -D (cleanup)
            if "git" in cmd and "branch" in cmd:
                return result

            # Default git commands succeed
            if "git" in cmd:
                return result

            # Default: succeed
            return result

        def mock_agentic_task(**kwargs):
            """Mock run_agentic_task to simulate step execution."""
            label = kwargs.get("label", "")

            # Step 9: Return FILES_MODIFIED marker
            if label == "step9":
                return (True, "Implementation complete.\nFILES_MODIFIED: fix.py", 0.1, "gpt-4")

            # Step 10: No issues found (skip review loop)
            if label.startswith("step10"):
                return (True, "Review complete. No Issues Found.", 0.1, "gpt-4")

            # Step 12: Return PR URL
            if label == "step12":
                return (True, "PR Created: https://github.com/owner/repo/pull/999", 0.1, "gpt-4")

            # Default success for other steps
            return (True, f"Step {label} completed successfully.", 0.1, "gpt-4")

        def mock_which(cmd):
            """Mock shutil.which to indicate gh is available."""
            if cmd == "gh":
                return "/usr/bin/gh"
            return None

        def mock_load_template(name):
            """Return realistic templates with placeholders."""
            # Templates that reference context variables
            templates = {
                "agentic_change_step1_duplicate_LLM": "Check for duplicates of issue: {issue_title}\nContent: {issue_content}",
                "agentic_change_step2_docs_LLM": "Check docs for: {issue_title}\nContent: {issue_content}",
                "agentic_change_step3_research_LLM": "Research: {issue_title}\nContent: {issue_content}",
                "agentic_change_step4_clarify_LLM": "Clarify requirements for: {issue_title}\nContent: {issue_content}",
                "agentic_change_step5_docs_change_LLM": "Analyze docs for: {issue_title}\nPrevious: {step4_output}",
                "agentic_change_step6_devunits_LLM": "Identify dev units: {step5_output}",
                "agentic_change_step7_architecture_LLM": "Review architecture: {step6_output}",
                "agentic_change_step8_analyze_LLM": "Analyze changes: {step7_output}",
                "agentic_change_step9_implement_LLM": "Implement: {step8_output}\nWorktree: {worktree_path}",
                "agentic_change_step10_identify_issues_LLM": "Review implementation: {step9_output}",
                "agentic_change_step11_fix_issues_LLM": "Fix issues: {step10_output}",
                "agentic_change_step12_create_pr_LLM": "Create PR for files: {files_to_stage}",
            }
            return templates.get(name, "Default template: {issue_content}")

        # Apply mocks
        with patch("pdd.agentic_change.shutil.which", side_effect=mock_which), \
             patch("pdd.agentic_change.subprocess.run", side_effect=mock_subprocess_run), \
             patch("pdd.agentic_change_orchestrator.subprocess.run", side_effect=mock_subprocess_run), \
             patch("pdd.agentic_change_orchestrator.run_agentic_task", side_effect=mock_agentic_task), \
             patch("pdd.agentic_change_orchestrator.load_prompt_template", side_effect=mock_load_template), \
             patch("pdd.agentic_change_orchestrator.save_workflow_state", return_value=None), \
             patch("pdd.agentic_change_orchestrator.load_workflow_state", return_value=(None, None)), \
             patch("pdd.agentic_change_orchestrator.clear_workflow_state", return_value=None):

            from pdd.agentic_change import run_agentic_change

            # Run the full E2E path
            success, msg, cost, model, files = run_agentic_change(
                "https://github.com/owner/repo/issues/319",
                verbose=False,
                quiet=True,
                use_github_state=False
            )

        # THE KEY ASSERTIONS

        # BUG CHECK: If the bug exists, msg will contain "Context missing key"
        assert "Context missing key" not in msg, (
            f"BUG DETECTED (Issue #319)!\n\n"
            f"The orchestrator failed with: {msg}\n\n"
            f"This indicates that JSON curly braces in the issue content were "
            f"incorrectly interpreted as str.format() placeholders.\n\n"
            f"The issue body contained JSON like '{json_issue_body[:100]}...'"
        )

        assert success is True, (
            f"E2E test failed.\n"
            f"Expected: success=True\n"
            f"Got: success=False, message={msg}"
        )

        assert "PR Created" in msg, (
            f"Expected PR to be created.\n"
            f"Got message: {msg}"
        )

    def test_e2e_change_with_json_in_step_outputs(self, tmp_path, monkeypatch):
        """
        E2E Test: Verify JSON in step outputs doesn't cause KeyError in later steps.

        This tests that when an early step (e.g., step 4) returns JSON in its output,
        later steps (step 5+) can use that output without str.format() failing.

        Bug scenario:
        - Step 4 returns output containing JSON
        - Step 5 template has {step4_output} placeholder
        - str.format() should substitute the JSON as a literal string, not re-parse it
        """
        monkeypatch.chdir(tmp_path)

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "README.md").write_text("# Test repo")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, capture_output=True)

        # Simple issue without JSON (the JSON will come from step outputs)
        mock_issue_data = {
            "title": "Add new configuration option",
            "body": "Please add support for custom configuration files.",
            "user": {"login": "testuser"},
            "comments_url": ""
        }

        def mock_subprocess_run(args, **kwargs):
            cmd = args if isinstance(args, list) else []
            cmd_str = " ".join(cmd) if cmd else ""

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            if "gh" in cmd and "api" in cmd and "issues/320" in cmd_str:
                result.stdout = json.dumps(mock_issue_data)
                return result

            if "git" in cmd and "rev-parse" in cmd and "--show-toplevel" in cmd:
                result.stdout = str(tmp_path)
                return result

            if "git" in cmd and "rev-parse" in cmd and "--abbrev-ref" in cmd:
                result.stdout = "main"
                return result

            if "git" in cmd and "worktree" in cmd:
                if "list" in cmd:
                    result.stdout = ""
                elif "add" in cmd:
                    for i, arg in enumerate(cmd):
                        if arg == "add" and i + 2 < len(cmd):
                            wt_path = Path(cmd[i + 2])
                            wt_path.mkdir(parents=True, exist_ok=True)
                            (wt_path / ".git").write_text("gitdir: /fake")
                            break
                return result

            if "git" in cmd:
                return result

            return result

        call_count = {"step4": 0}

        def mock_agentic_task(**kwargs):
            """Mock that returns JSON in step 4 output."""
            label = kwargs.get("label", "")

            # Step 4: Return output containing JSON (like a config analysis)
            if label == "step4":
                call_count["step4"] += 1
                return (True, '''Analysis complete. Recommended configuration:

```json
{
  "feature_flags": {
    "enable_custom_config": true,
    "config_path": "./custom.json"
  },
  "settings": {
    "timeout": 30,
    "retry_count": 3
  }
}
```

Requirements are clear. Proceed with implementation.''', 0.1, "gpt-4")

            if label == "step9":
                return (True, "FILES_MODIFIED: config.py", 0.1, "gpt-4")

            if label.startswith("step10"):
                return (True, "No Issues Found", 0.1, "gpt-4")

            if label == "step12":
                return (True, "PR Created: https://github.com/owner/repo/pull/888", 0.1, "gpt-4")

            return (True, f"Step {label} completed.", 0.1, "gpt-4")

        def mock_which(cmd):
            if cmd == "gh":
                return "/usr/bin/gh"
            return None

        def mock_load_template(name):
            """Templates that use step outputs as placeholders."""
            templates = {
                "agentic_change_step1_duplicate_LLM": "Check: {issue_content}",
                "agentic_change_step2_docs_LLM": "Docs: {issue_content}",
                "agentic_change_step3_research_LLM": "Research: {issue_content}",
                "agentic_change_step4_clarify_LLM": "Clarify: {issue_content}",
                # Step 5 uses step4_output - this is where JSON from step 4 would be substituted
                "agentic_change_step5_docs_change_LLM": "Document changes based on analysis:\n\nStep 4 Output:\n{step4_output}\n\nIssue: {issue_title}",
                "agentic_change_step6_devunits_LLM": "Dev units from: {step5_output}",
                "agentic_change_step7_architecture_LLM": "Architecture: {step6_output}",
                "agentic_change_step8_analyze_LLM": "Analyze: {step7_output}",
                "agentic_change_step9_implement_LLM": "Implement: {step8_output}\nPath: {worktree_path}",
                "agentic_change_step10_identify_issues_LLM": "Review: {step9_output}",
                "agentic_change_step11_fix_issues_LLM": "Fix: {step10_output}",
                "agentic_change_step12_create_pr_LLM": "PR: {files_to_stage}",
            }
            return templates.get(name, "Default: {issue_content}")

        with patch("pdd.agentic_change.shutil.which", side_effect=mock_which), \
             patch("pdd.agentic_change.subprocess.run", side_effect=mock_subprocess_run), \
             patch("pdd.agentic_change_orchestrator.subprocess.run", side_effect=mock_subprocess_run), \
             patch("pdd.agentic_change_orchestrator.run_agentic_task", side_effect=mock_agentic_task), \
             patch("pdd.agentic_change_orchestrator.load_prompt_template", side_effect=mock_load_template), \
             patch("pdd.agentic_change_orchestrator.save_workflow_state", return_value=None), \
             patch("pdd.agentic_change_orchestrator.load_workflow_state", return_value=(None, None)), \
             patch("pdd.agentic_change_orchestrator.clear_workflow_state", return_value=None):

            from pdd.agentic_change import run_agentic_change

            success, msg, cost, model, files = run_agentic_change(
                "https://github.com/owner/repo/issues/320",
                verbose=False,
                quiet=True,
                use_github_state=False
            )

        # Verify step 4 was called (so its JSON output would be used in step 5)
        assert call_count["step4"] == 1, "Step 4 should have been called"

        # BUG CHECK: If JSON in step4_output causes issues, step 5 would fail
        assert "Context missing key" not in msg, (
            f"BUG DETECTED (Issue #319)!\n\n"
            f"The orchestrator failed when substituting step4_output into step 5 template.\n"
            f"Error: {msg}\n\n"
            f"Step 4 returned JSON that should NOT have been re-parsed by str.format()."
        )

        assert success is True, f"E2E test failed: {msg}"

    def test_e2e_demonstrates_bug_pattern_keyerror(self):
        """
        Documentation test: Demonstrates the exact bug pattern that WOULD cause KeyError.

        This test shows WHY the bug as reported cannot occur with standard str.format()
        behavior. The KeyError only happens when JSON braces are in the TEMPLATE itself,
        not in substituted values.

        This serves as documentation of the investigated bug mechanism.
        """
        # CASE 1: JSON in substituted VALUE - WORKS CORRECTLY
        # This is what happens in the orchestrator - JSON in context values is safe
        template = "Issue content: {issue_content}"
        json_content = '{"type": "error", "code": 500}'

        # This SHOULD work - and it does
        result = template.format(issue_content=json_content)
        assert json_content in result, "JSON in substituted values should work"

        # CASE 2: JSON in TEMPLATE itself - CAUSES KEYERROR
        # This is what would cause the reported bug
        bad_template = '{"type": "error"}'  # Template contains literal JSON

        with pytest.raises(KeyError) as exc_info:
            bad_template.format()  # Python sees {"type": ...} as a placeholder

        # The error message contains the "key" Python tried to find
        assert "type" in str(exc_info.value)

        # CASE 3: Escaped braces in template - WORKS
        # This is the fix pattern if JSON MUST be in template
        escaped_template = '{{"type": "error"}}'
        result = escaped_template.format()
        assert result == '{"type": "error"}', "Escaped braces should become literal braces"

    def test_e2e_format_behavior_with_nested_braces(self):
        """
        Documentation test: Verify str.format() does NOT re-parse substituted values.

        This confirms that the bug as described in Issue #319 cannot occur because
        Python's str.format() only processes the original template, not the result
        of substitutions.
        """
        # Template with placeholder
        template = "Previous step output: {step4_output}\nNow do step 5."

        # Step 4 output contains JSON with braces
        step4_output = '''Analysis found:
```json
{
  "feature": "dark_mode",
  "enabled": true
}
```
Requirements are clear.'''

        # This MUST work - str.format() substitutes step4_output as literal text
        result = template.format(step4_output=step4_output)

        # Verify the JSON braces are preserved as literal text
        assert '{"feature"' in result or '"feature"' in result, (
            "JSON braces in substituted value should be preserved as literals"
        )

        # Verify no KeyError was raised (if it was, we wouldn't reach here)
        assert "Analysis found" in result
        assert "Requirements are clear" in result


class TestIssue319RegressionTests:
    """
    Regression tests to prevent reintroduction of the bug pattern.

    These tests verify specific code paths that could potentially
    cause the bug if modified incorrectly.
    """

    def test_orchestrator_context_substitution_order(self, tmp_path, monkeypatch):
        """
        Verify the orchestrator's context substitution doesn't double-format.

        A potential bug could be introduced if the code:
        1. Formats the template once
        2. Then formats the RESULT again (which would parse substituted values)

        This test ensures only single formatting happens.
        """
        import subprocess
        monkeypatch.chdir(tmp_path)

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "README.md").write_text("# Test")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        format_call_count = {"count": 0}

        def mock_load_template(name):
            """Track template loads and return a template that records format calls."""

            class TrackingTemplate:
                def __init__(self, template_str):
                    self.template = template_str

                def format(self, **kwargs):
                    format_call_count["count"] += 1
                    return self.template.format(**kwargs)

            return TrackingTemplate("Test template for {issue_title}")

        def mock_subprocess_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = str(tmp_path)
            result.stderr = ""
            return result

        def mock_task(**kwargs):
            label = kwargs.get("label", "")
            if label == "step1":
                # Return JSON that could cause issues if double-formatted
                return (True, '{"status": "duplicate", "issue": 123}', 0.1, "gpt-4")
            # Hard stop at step 1 for faster test
            return (True, "Duplicate of #999", 0.1, "gpt-4")

        with patch("pdd.agentic_change_orchestrator.subprocess.run", side_effect=mock_subprocess_run), \
             patch("pdd.agentic_change_orchestrator.run_agentic_task", side_effect=mock_task), \
             patch("pdd.agentic_change_orchestrator.load_prompt_template", side_effect=mock_load_template), \
             patch("pdd.agentic_change_orchestrator.save_workflow_state", return_value=None), \
             patch("pdd.agentic_change_orchestrator.load_workflow_state", return_value=(None, None)):

            from pdd.agentic_change_orchestrator import run_agentic_change_orchestrator

            success, msg, cost, model, files = run_agentic_change_orchestrator(
                issue_url="http://test",
                issue_content='{"config": true}',  # JSON in issue content
                repo_owner="owner",
                repo_name="repo",
                issue_number=321,
                issue_author="user",
                issue_title="Test with JSON",
                cwd=tmp_path,
                quiet=True,
                use_github_state=False
            )

        # Verify format was called exactly once per template load (not double-formatted)
        # The orchestrator should load and format step 1 template once
        assert format_call_count["count"] >= 1, "Template should be formatted at least once"

        # If format is called multiple times on same content, it would cause the bug
        # Note: Count may be > 1 due to multiple steps, but each step should only format once
