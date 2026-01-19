# TEST PLAN
#
# 1. Unit Tests (Pytest):
#    - test_orchestrator_happy_path: Verifies the complete 1-12 step flow with successful execution.
#      Checks that cost accumulates, files are parsed at step 9, and PR URL is extracted at step 12.
#    - test_orchestrator_hard_stop_early: Verifies that the orchestrator stops immediately if a hard stop
#      condition is met (e.g., "Duplicate of #" in Step 1).
#    - test_orchestrator_resume_from_state: Verifies that if a state file exists, previously completed
#      steps are skipped and execution resumes from the correct step.
#    - test_orchestrator_worktree_failure: Verifies behavior when git worktree setup fails (should return False).
#    - test_orchestrator_step9_failure_no_files: Verifies failure at Step 9 if no files are parsed from output.
#    - test_orchestrator_review_loop_logic: Verifies the interaction between Step 10 and 11.
#      Scenario: Step 10 finds issues -> Step 11 fixes -> Step 10 finds no issues -> Proceed.
#    - test_orchestrator_review_loop_max_iterations: Verifies that the loop breaks after MAX_REVIEW_ITERATIONS
#      even if issues persist.
#
# 2. Z3 Formal Verification:
#    - test_z3_review_loop_termination: Models the review loop logic (Steps 10-11) as a state machine
#      to formally prove that the loop is guaranteed to terminate either by finding no issues or hitting
#      the iteration limit.

import json
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
from z3 import Solver, Int, Bool, Implies, And, Or, Not, unsat

# Adjust import path to ensure we can import the module under test
from pdd.agentic_change_orchestrator import run_agentic_change_orchestrator

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_cwd(tmp_path):
    """Returns a temporary directory path to use as cwd."""
    return tmp_path

@pytest.fixture
def mock_dependencies(temp_cwd):
    """
    Mocks the external dependencies: run_agentic_task, load_prompt_template,
    and subprocess (for git operations).
    """
    with patch("pdd.agentic_change_orchestrator.run_agentic_task") as mock_run, \
         patch("pdd.agentic_change_orchestrator.load_prompt_template") as mock_load, \
         patch("pdd.agentic_change_orchestrator.subprocess.run") as mock_subprocess, \
         patch("pdd.agentic_change_orchestrator.console") as mock_console:
        
        # Default mock behaviors
        mock_run.return_value = (True, "Default Agent Output", 0.1, "gpt-4")
        
        mock_template = MagicMock()
        mock_template.format.return_value = "Formatted Prompt"
        mock_load.return_value = mock_template
        
        # Mock git rev-parse to return the temp_cwd as root
        # This ensures mkdir operations on the root succeed
        mock_subprocess.return_value.stdout = str(temp_cwd)
        mock_subprocess.return_value.returncode = 0
        
        yield mock_run, mock_load, mock_subprocess, mock_console

# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

def test_orchestrator_happy_path(mock_dependencies, temp_cwd):
    """
    Test the full successful execution of the orchestrator (Steps 1-12).
    """
    mock_run, mock_load, mock_subprocess, _ = mock_dependencies

    # Setup specific outputs for key steps
    # Note: Review loop uses step10_iter1, step11_iter1 labels
    def side_effect_run(**kwargs):
        label = kwargs.get("label", "")
        if label == "step9":
            return (True, "Implementation done. FILES_MODIFIED: file_a.py, file_b.py", 0.5, "gpt-4")
        if label.startswith("step10"):
            return (True, "No Issues Found", 0.1, "gpt-4")
        if label == "step12":
            return (True, "PR Created: https://github.com/owner/repo/pull/123", 0.2, "gpt-4")
        return (True, f"Output for {label}", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    success, msg, cost, model, files = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="Fix bug",
        repo_owner="owner",
        repo_name="repo",
        issue_number=1,
        issue_author="me",
        issue_title="Bug fix",
        cwd=temp_cwd,
        verbose=True
    )

    assert success is True
    assert "PR Created: https://github.com/owner/repo/pull/123" in msg
    assert "file_a.py" in files
    assert "file_b.py" in files
    # Cost calculation: 
    # Steps 1-8 (8 * 0.1) + Step 9 (0.5) + Step 10 (0.1) + Step 12 (0.2) = 0.8 + 0.5 + 0.1 + 0.2 = 1.6
    assert cost == pytest.approx(1.6)
    
    # Verify state file was cleared
    state_file = temp_cwd / ".pdd/change-state/change_state_1.json"
    assert not state_file.exists()

def test_orchestrator_hard_stop_early(mock_dependencies, temp_cwd):
    """
    Test that the orchestrator stops immediately if a hard stop condition is met.
    """
    mock_run, _, _, _ = mock_dependencies

    # Step 1 returns "Duplicate of #"
    mock_run.return_value = (True, "This is a Duplicate of #42", 0.1, "gpt-4")

    success, msg, cost, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="Fix bug",
        repo_owner="owner",
        repo_name="repo",
        issue_number=2,
        issue_author="me",
        issue_title="Duplicate",
        cwd=temp_cwd
    )

    assert success is False
    assert "Stopped at step 1" in msg
    assert "Issue is a duplicate" in msg
    assert cost == 0.1
    
    # Verify state file exists (persisted on stop)
    state_file = temp_cwd / ".pdd/change-state/change_state_2.json"
    assert state_file.exists()

def test_orchestrator_resume_from_state(mock_dependencies, temp_cwd):
    """
    Test resumption from a saved state file.
    """
    mock_run, _, _, _ = mock_dependencies

    # Create a state file simulating completion of steps 1-4
    state_dir = temp_cwd / ".pdd/change-state"
    state_dir.mkdir(parents=True)
    state_file = state_dir / "change_state_3.json"
    
    initial_state = {
        "issue_number": 3,
        "last_completed_step": 4,
        "step_outputs": {
            "1": "out1", "2": "out2", "3": "out3", "4": "out4"
        },
        "total_cost": 1.0,
        "model_used": "gpt-3.5"
    }
    with open(state_file, "w") as f:
        json.dump(initial_state, f)

    # Mock subsequent steps
    # Note: Review loop uses step10_iter1, step11_iter1 labels
    def side_effect_run(**kwargs):
        label = kwargs.get("label", "")
        if label == "step9":
            return (True, "FILES_CREATED: new.py", 0.5, "gpt-4")
        if label.startswith("step10"):
            return (True, "No Issues Found", 0.1, "gpt-4")
        if label == "step12":
            return (True, "PR Created", 0.1, "gpt-4")
        return (True, "ok", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    success, _, cost, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="content",
        repo_owner="owner",
        repo_name="repo",
        issue_number=3,
        issue_author="me",
        issue_title="Resume",
        cwd=temp_cwd
    )

    assert success is True
    # Verify steps 1-4 were NOT called
    labels_called = [call.kwargs.get('label') for call in mock_run.call_args_list]
    assert "step1" not in labels_called
    assert "step4" not in labels_called
    assert "step5" in labels_called

    # Initial cost 1.0 + steps 5,6,7,8 (0.4) + step 9 (0.5) + step 10 (0.1) + step 12 (0.1) = 2.1
    assert cost == pytest.approx(2.1)

def test_orchestrator_worktree_failure(mock_dependencies, temp_cwd):
    """
    Test failure when setting up the git worktree.
    """
    mock_run, _, mock_subprocess, _ = mock_dependencies

    def side_effect_subprocess(args, **kwargs):
        # Simulate failure for worktree add
        if "worktree" in args and "add" in args:
            raise subprocess.CalledProcessError(1, args, stderr="Worktree creation failed")
        
        # Simulate success for rev-parse (returning temp_cwd as root)
        mock_ret = MagicMock()
        mock_ret.returncode = 0
        mock_ret.stdout = str(temp_cwd)
        return mock_ret

    mock_subprocess.side_effect = side_effect_subprocess

    # Mock steps 1-8 to pass
    mock_run.return_value = (True, "ok", 0.1, "gpt-4")

    success, msg, _, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="content",
        repo_owner="owner",
        repo_name="repo",
        issue_number=4,
        issue_author="me",
        issue_title="Worktree Fail",
        cwd=temp_cwd
    )

    assert success is False
    assert "Failed to create worktree" in msg

def test_orchestrator_step9_failure_no_files(mock_dependencies, temp_cwd):
    """
    Test failure at Step 9 if no files are detected in the output.
    """
    mock_run, _, _, _ = mock_dependencies

    def side_effect_run(**kwargs):
        label = kwargs.get("label", "")
        if label == "step9":
            return (True, "I implemented it but forgot to list files.", 0.5, "gpt-4")
        return (True, "ok", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    success, msg, _, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="content",
        repo_owner="owner",
        repo_name="repo",
        issue_number=5,
        issue_author="me",
        issue_title="Step 9 Fail",
        cwd=temp_cwd
    )

    assert success is False
    assert "Stopped at step 9" in msg
    assert "no file changes" in msg

def test_orchestrator_step9_failure_preserves_completed_steps(mock_dependencies, temp_cwd):
    """
    Test that when step 9 fails, state correctly shows steps 6-8 as completed.

    Bug scenario:
    1. State loaded with last_completed_step=5
    2. Steps 6, 7, 8 succeed
    3. Step 9 triggers worktree setup, then fails (no FILES markers)
    4. State should have last_completed_step=8, NOT 5

    This catches a bug where line 313's state save uses a stale variable
    instead of step_num - 1, causing progress from steps 6-8 to be lost.
    """
    mock_run, _, _, _ = mock_dependencies

    # Create initial state with steps 1-5 completed
    state_dir = temp_cwd / ".pdd/change-state"
    state_dir.mkdir(parents=True)
    state_file = state_dir / "change_state_99.json"

    initial_state = {
        "issue_number": 99,
        "last_completed_step": 5,
        "step_outputs": {"1": "o1", "2": "o2", "3": "o3", "4": "o4", "5": "o5"},
        "total_cost": 0.5,
        "model_used": "gpt-4"
    }
    with open(state_file, "w") as f:
        json.dump(initial_state, f)

    def side_effect_run(**kwargs):
        label = kwargs.get("label", "")
        if label == "step9":
            # Return output WITHOUT FILES markers - triggers failure
            return (True, "I did the work but no FILES_CREATED marker", 0.1, "gpt-4")
        return (True, f"Output for {label}", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    success, msg, _, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="content",
        repo_owner="owner",
        repo_name="repo",
        issue_number=99,
        issue_author="me",
        issue_title="State Bug",
        cwd=temp_cwd
    )

    assert success is False
    assert "step 9" in msg.lower()

    # CRITICAL: Verify state was saved with correct last_completed_step
    with open(state_file, "r") as f:
        final_state = json.load(f)

    # Steps 6, 7, 8 completed successfully before step 9 failed
    assert final_state["last_completed_step"] == 8, \
        f"Expected last_completed_step=8, got {final_state['last_completed_step']}"

    # Verify step outputs exist for 6, 7, 8
    assert "6" in final_state["step_outputs"]
    assert "7" in final_state["step_outputs"]
    assert "8" in final_state["step_outputs"]

def test_orchestrator_review_loop_logic(mock_dependencies, temp_cwd):
    """
    Test the review loop: Step 10 finds issues -> Step 11 fixes -> Step 10 finds no issues.
    Note: Review loop uses step10_iter1, step11_iter1, step10_iter2, etc. labels.
    """
    mock_run, _, _, _ = mock_dependencies

    step10_calls = 0

    def side_effect_run(**kwargs):
        nonlocal step10_calls
        label = kwargs.get("label", "")

        if label == "step9":
            return (True, "FILES_MODIFIED: a.py", 0.1, "gpt-4")
        elif label.startswith("step10"):
            step10_calls += 1
            if step10_calls == 1:
                return (True, "Issues Found: Bad style", 0.1, "gpt-4")
            else:
                return (True, "No Issues Found", 0.1, "gpt-4")
        elif label.startswith("step11"):
            return (True, "Fixed style", 0.1, "gpt-4")
        elif label == "step12":
            return (True, "PR Created", 0.1, "gpt-4")
        return (True, "ok", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    success, _, _, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="content",
        repo_owner="owner",
        repo_name="repo",
        issue_number=6,
        issue_author="me",
        issue_title="Review Loop",
        cwd=temp_cwd
    )

    assert success is True
    assert step10_calls == 2

    step11_calls = [call for call in mock_run.call_args_list if call.kwargs.get('label', '').startswith('step11')]
    assert len(step11_calls) == 1

def test_orchestrator_review_loop_max_iterations(mock_dependencies, temp_cwd):
    """
    Test that the review loop terminates after max iterations even if issues persist.
    Note: Review loop uses step10_iterN, step11_iterN labels.
    """
    mock_run, _, _, _ = mock_dependencies

    def side_effect_run(**kwargs):
        label = kwargs.get("label", "")
        if label == "step9":
            return (True, "FILES_MODIFIED: a.py", 0.1, "gpt-4")
        elif label.startswith("step10"):
            return (True, "Issues Found: Still broken", 0.1, "gpt-4")
        elif label.startswith("step11"):
            return (True, "Attempted fix", 0.1, "gpt-4")
        elif label == "step12":
            return (True, "PR Created", 0.1, "gpt-4")
        return (True, "ok", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    success, _, _, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="content",
        repo_owner="owner",
        repo_name="repo",
        issue_number=7,
        issue_author="me",
        issue_title="Max Iterations",
        cwd=temp_cwd
    )

    assert success is True
    step10_calls = [call for call in mock_run.call_args_list if call.kwargs.get('label', '').startswith('step10')]
    assert len(step10_calls) == 5

    step11_calls = [call for call in mock_run.call_args_list if call.kwargs.get('label', '').startswith('step11')]
    assert len(step11_calls) == 5

# -----------------------------------------------------------------------------
# Step 7 Stop Condition Tests (TDD)
# -----------------------------------------------------------------------------

def test_step7_stop_with_stop_condition_marker(mock_dependencies, temp_cwd):
    """
    Test that Step 7 stops when explicit stop condition is present.

    Implementation checks for exact string "Architectural Decision Needed" (case-sensitive).
    """
    mock_run, _, _, _ = mock_dependencies

    def side_effect(**kwargs):
        label = kwargs.get("label", "")
        if label == "step7":
            # Use exact case that implementation checks for
            return (True, "Posted to GitHub.\nArchitectural Decision Needed", 0.1, "gpt-4")
        return (True, f"Output for {label}", 0.1, "gpt-4")

    mock_run.side_effect = side_effect

    success, msg, _, _, _ = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content="Feature request",
        repo_owner="owner",
        repo_name="repo",
        issue_number=777,
        issue_author="user",
        issue_title="Feature",
        cwd=temp_cwd
    )

    assert success is False, "Workflow should have stopped at step 7"
    assert "Stopped at step 7" in msg
    assert "Architectural decision needed" in msg


def test_step7_prompt_has_stop_condition_marker():
    """
    Verify Step 7 prompt documents the exact STOP_CONDITION marker.

    TDD: This test FAILS until we add the CRITICAL section to the prompt.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "agentic_change_step7_architecture_LLM.prompt"
    prompt_content = prompt_path.read_text()

    # Must have CRITICAL section
    assert "% CRITICAL" in prompt_content, "Step 7 prompt missing CRITICAL section"

    # Must document exact marker
    assert "STOP_CONDITION: Architectural decision needed" in prompt_content, (
        "Step 7 prompt must document exact marker: 'STOP_CONDITION: Architectural decision needed'"
    )


# -----------------------------------------------------------------------------
# Scope Enforcement Tests (TDD for PDD Methodology)
# -----------------------------------------------------------------------------

def test_step9_prompt_has_scope_critical_section():
    """
    Verify Step 9 prompt has CRITICAL scope section prominently placed.

    TDD: This test FAILS until we update the prompt with:
    - % CRITICAL: Scope section
    - FORBIDDEN keyword
    - References to Code files and Example files
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "agentic_change_step9_implement_LLM.prompt"
    prompt_content = prompt_path.read_text()

    # Must have CRITICAL scope section
    assert "% CRITICAL: Scope" in prompt_content, "Step 9 prompt missing % CRITICAL: Scope section"

    # Must forbid code/example files
    assert "FORBIDDEN" in prompt_content, "Step 9 prompt must use FORBIDDEN keyword"
    assert "Code files" in prompt_content or "code files" in prompt_content, \
        "Step 9 prompt must mention Code files as forbidden"
    assert "Example files" in prompt_content or "example files" in prompt_content, \
        "Step 9 prompt must mention Example files as forbidden"


def test_step8_prompt_has_scope_section():
    """
    Verify Step 8 prompt has scope constraints.

    TDD: This test FAILS until we add a Scope section to Step 8.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "agentic_change_step8_analyze_LLM.prompt"
    prompt_content = prompt_path.read_text()

    # Must have scope section
    assert "% Scope" in prompt_content, "Step 8 prompt missing % Scope section"

    # Must mention what NOT to do
    assert "Do NOT" in prompt_content, "Step 8 prompt must say what NOT to do"
    assert "Code files" in prompt_content or "code files" in prompt_content, \
        "Step 8 prompt must mention Code files as forbidden"


def test_step6_prompt_clarifies_change_scope():
    """
    Verify Step 6 clarifies that pdd change only modifies prompts.

    TDD: This test FAILS until we add scope clarification to Step 6.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "agentic_change_step6_devunits_LLM.prompt"
    prompt_content = prompt_path.read_text()

    # Must clarify scope - pdd change only modifies prompts
    assert "pdd change" in prompt_content and "ONLY" in prompt_content, \
        "Step 6 must clarify that pdd change modifies ONLY prompts"
    assert "GENERATED" in prompt_content, \
        "Step 6 must clarify that code/example files are GENERATED"


# -----------------------------------------------------------------------------
# Z3 Formal Verification
# -----------------------------------------------------------------------------

def test_z3_review_loop_termination():
    """
    Formally verify that the review loop logic terminates.
    """
    s = Solver()
    MAX_ITERATIONS = 5
    
    def get_state(k):
        iteration = Int(f"iter_{k}")
        terminated = Bool(f"term_{k}")
        issues_found = Bool(f"issues_{k}") 
        return iteration, terminated, issues_found

    iter_0, term_0, _ = get_state(0)
    s.add(iter_0 == 0)
    s.add(term_0 == False)
    
    for k in range(MAX_ITERATIONS):
        iter_k, term_k, issues_found_k = get_state(k)
        iter_next, term_next, _ = get_state(k + 1)
        
        new_iter = iter_k + 1
        
        transition = Implies(
            Not(term_k),
            And(
                iter_next == new_iter,
                term_next == Or(
                    Not(issues_found_k),
                    new_iter >= MAX_ITERATIONS
                )
            )
        )
        
        persist = Implies(
            term_k,
            And(iter_next == iter_k, term_next == True)
        )
        
        s.add(And(transition, persist))

    _, term_final, _ = get_state(MAX_ITERATIONS)
    s.add(Not(term_final))

    result = s.check()
    assert result == unsat, "The review loop is not guaranteed to terminate within MAX_ITERATIONS"


# -----------------------------------------------------------------------------
# Prompt Template Tests
# -----------------------------------------------------------------------------

def test_step9_prompt_template_includes_step5_output():
    """
    TDD test: Verify Step 9 prompt template references step5_output.

    The orchestrator already includes step5_output in context (lines 270-272),
    but the template must actually reference {step5_output} for the agent to see it.

    Python's str.format(**context) silently ignores extra context keys,
    so missing {step5_output} in the template means documentation changes
    from Step 5 are never shown to the implementation agent.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "agentic_change_step9_implement_LLM.prompt"
    template_content = prompt_path.read_text()

    assert "{step5_output}" in template_content, \
        "Step 9 template must include {step5_output} to receive documentation changes from Step 5"


# -----------------------------------------------------------------------------
# Curly Brace / JSON Handling Tests (Issue #319)
# -----------------------------------------------------------------------------

def test_format_with_json_curly_braces_causes_keyerror():
    """
    Documents the exact bug pattern: str.format() interprets JSON braces as placeholders.

    When a string containing JSON like '{"type": "test"}' is passed through str.format(),
    Python interprets the curly braces as format placeholders and raises KeyError.

    This test documents the failure mode - it passes by demonstrating the error.
    """
    json_content = '{\n  "type": "test"\n}'
    template = "Issue content: {issue_content}"

    # This shows why the bug occurs - str.format will fail if JSON is in the template itself
    bad_template = f"Issue content: {json_content}"

    with pytest.raises(KeyError) as exc_info:
        bad_template.format()  # No placeholders needed - the JSON itself causes the error

    # The error shows Python trying to interpret the JSON key as a placeholder
    assert "type" in str(exc_info.value)


def test_escape_braces_in_template():
    """
    Test that properly escaped braces are not interpreted as format placeholders.

    This demonstrates the fix: doubling curly braces escapes them in str.format().
    """
    # When braces are properly escaped (doubled), str.format() treats them as literal braces
    escaped_json = '{{\n  "type": "test"\n}}'
    template_with_escaped = f"Issue content: {escaped_json}"

    # This should NOT raise - escaped braces are treated as literals
    result = template_with_escaped.format()

    # Result should have single braces (escaped braces become single)
    assert '{\n  "type": "test"\n}' in result


def test_orchestrator_handles_json_in_issue_content(mock_dependencies, temp_cwd):
    """
    Test that the orchestrator can handle JSON content in the issue body.

    Bug scenario (Issue #319): When the GitHub issue contains JSON code blocks,
    the curly braces cause str.format() to fail with KeyError.

    This test requires the template's format() to actually be called with the real
    context values, not a mock that ignores them.

    This test FAILS on buggy code and PASSES once the fix is applied.
    """
    mock_run, mock_load, _, _ = mock_dependencies

    # Issue content with JSON that contains curly braces
    json_issue_content = '''
    The user reported this error:
    ```json
    {
      "type": "error",
      "message": "Connection failed",
      "code": 500
    }
    ```
    Please fix this bug.
    '''

    # Create a realistic template that has actual placeholders
    # When the template is formatted, if issue_content contains unescaped braces,
    # and those braces are somehow incorporated into the template BEFORE format(),
    # it would cause a KeyError
    realistic_template = "Issue: {issue_content}\nURL: {issue_url}"

    # Make load_prompt_template return the realistic template string
    # so format(**context) is actually called on it
    mock_load.return_value = realistic_template

    # Configure mock to return success for all steps
    def side_effect_run(**kwargs):
        label = kwargs.get("label", "")
        if label == "step9":
            return (True, "FILES_MODIFIED: fix.py", 0.1, "gpt-4")
        if label.startswith("step10"):
            return (True, "No Issues Found", 0.1, "gpt-4")
        if label == "step12":
            return (True, "PR Created: https://github.com/owner/repo/pull/1", 0.1, "gpt-4")
        return (True, f"Output for {label}", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    # The orchestrator should handle JSON in issue content without KeyError
    success, msg, cost, model, files = run_agentic_change_orchestrator(
        issue_url="http://url",
        issue_content=json_issue_content,  # Contains JSON with curly braces
        repo_owner="owner",
        repo_name="repo",
        issue_number=319,
        issue_author="user",
        issue_title="Bug with JSON",
        cwd=temp_cwd,
        quiet=True
    )

    # If the bug exists, this will fail with:
    # "Context missing key for step 1: 'type'" or similar
    assert "Context missing key" not in msg, f"Bug detected: {msg}"
    assert success is True, f"Orchestrator failed: {msg}"


def test_orchestrator_handles_curly_braces_in_step_outputs(mock_dependencies, temp_cwd):
    """
    Test that the orchestrator handles curly braces in step outputs.

    Bug scenario: When a step returns output containing JSON/curly braces,
    subsequent steps may fail because str.format() interprets those braces.

    This test uses a realistic template with {step4_output} placeholder to
    verify the context substitution works even when step outputs contain JSON.

    This test FAILS on buggy code and PASSES once the fix is applied.
    """
    mock_run, _, mock_subprocess, _ = mock_dependencies

    call_count = 0

    def side_effect_run(**kwargs):
        nonlocal call_count
        call_count += 1
        label = kwargs.get("label", "")

        # Step 4 returns JSON-containing output (like a code analysis result)
        if label == "step4":
            return (True, '''Analysis complete. Found configuration:
```json
{
  "feature": "dark_mode",
  "enabled": true
}
```
Requirements are clear.''', 0.1, "gpt-4")

        if label == "step9":
            return (True, "FILES_MODIFIED: config.py", 0.1, "gpt-4")
        if label.startswith("step10"):
            return (True, "No Issues Found", 0.1, "gpt-4")
        if label == "step12":
            return (True, "PR Created: https://github.com/owner/repo/pull/2", 0.1, "gpt-4")
        return (True, f"Output for {label}", 0.1, "gpt-4")

    mock_run.side_effect = side_effect_run

    # Use realistic templates that include step output placeholders
    # The step 5+ templates reference {step4_output} etc.
    templates = {
        "agentic_change_step1_duplicate_LLM": "Check issue: {issue_content}",
        "agentic_change_step2_docs_LLM": "Check docs for: {issue_content}",
        "agentic_change_step3_research_LLM": "Research: {issue_content}",
        "agentic_change_step4_clarify_LLM": "Clarify: {issue_content}",
        "agentic_change_step5_docs_change_LLM": "Docs for: {issue_content}\nStep4: {step4_output}",
        "agentic_change_step6_devunits_LLM": "Dev units: {step4_output}\n{step5_output}",
        "agentic_change_step7_architecture_LLM": "Arch: {step5_output}",
        "agentic_change_step8_analyze_LLM": "Analyze: {step6_output}",
        "agentic_change_step9_implement_LLM": "Implement: {step8_output}",
        "agentic_change_step10_identify_issues_LLM": "Review: {step9_output}",
        "agentic_change_step11_fix_issues_LLM": "Fix: {step10_output}",
        "agentic_change_step12_create_pr_LLM": "Create PR: {step9_output}",
    }

    def mock_load_template(name):
        return templates.get(name, "Default: {issue_content}")

    with patch("pdd.agentic_change_orchestrator.load_prompt_template") as mock_load:
        mock_load.side_effect = mock_load_template

        success, msg, cost, model, files = run_agentic_change_orchestrator(
            issue_url="http://url",
            issue_content="Add dark mode feature",
            repo_owner="owner",
            repo_name="repo",
            issue_number=320,
            issue_author="user",
            issue_title="Add dark mode",
            cwd=temp_cwd,
            quiet=True
        )

    # If the bug exists, step 5+ will fail trying to format step4_output's JSON
    assert "Context missing key" not in msg, f"Bug detected: {msg}"
    assert success is True, f"Orchestrator failed: {msg}"

    # Verify we actually got past step 4 (steps 5-12 should have been called)
    labels_called = [call.kwargs.get('label') for call in mock_run.call_args_list]
    assert "step5" in labels_called, "Step 5 was never called - test may not be exercising the bug path"