# E2E Test for GitHub Issue #302: Pydantic serialization warnings from LiteLLM
# https://github.com/promptdriven/pdd/issues/302
#
# This E2E test verifies that PDD CLI commands do not emit Pydantic serialization
# warnings to users when LiteLLM's caching layer serializes ModelResponse objects.
#
# Bug Summary:
# - LiteLLM's response models (Message, Choices, etc.) delete optional attributes
#   after initialization to match OpenAI's API spec
# - When Pydantic serializes these objects during caching, it triggers
#   PydanticSerializationUnexpectedValue warnings
# - These warnings are cosmetic but clutter user output
#
# E2E Test Strategy:
# - Use Click's CliRunner to invoke real PDD CLI commands
# - Mock only the final LLM API call (to avoid costs), not caching/serialization
# - Capture stderr/warnings and verify no PydanticSerializationUnexpectedValue messages
# - Test multiple PDD commands that use LLM invocation (generate, test, crash)

import pytest
import warnings
import sys
import io
from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def set_pdd_path(monkeypatch):
    """Set PDD_PATH to the pdd package directory for all tests in this module.

    This is required because construct_paths uses PDD_PATH to find the language_format.csv
    file for language detection.
    """
    import pdd
    pdd_package_dir = Path(pdd.__file__).parent
    monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))


def create_mock_litellm_response():
    """Create a mock LiteLLM response that mimics real ModelResponse structure.

    This creates a response that would trigger Pydantic serialization warnings
    if not properly suppressed, because it has the same structure as real
    LiteLLM responses.
    """
    from litellm.types.utils import ModelResponse, Message, Choices, Usage

    # Create actual LiteLLM types to trigger the serialization behavior
    msg = Message(content='{"result": "test output"}', role="assistant")
    choice = Choices(finish_reason="stop", index=0, message=msg)
    usage = Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)

    response = ModelResponse(
        id="test-response-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=usage
    )

    return response


class TestE2EPydanticWarningsCliGenerate:
    """E2E tests for Pydantic warning suppression in pdd generate command."""

    def test_generate_command_does_not_emit_pydantic_warnings(self, tmp_path, monkeypatch):
        """
        E2E TEST: pdd generate command should NOT emit Pydantic serialization warnings.

        This test:
        1. Creates real prompt and code files
        2. Runs pdd generate via CliRunner (real CLI flow)
        3. Mocks only the LiteLLM completion call
        4. Captures warnings and verifies none are Pydantic serialization warnings

        EXPECTED TO FAIL until the fix is applied.
        """
        # Set up test files
        prompt_content = """Generate a simple hello world function.

Expected output:
```python
def hello():
    return "Hello, World!"
```
"""
        (tmp_path / "hello.prompt").write_text(prompt_content)

        monkeypatch.chdir(tmp_path)

        # Create mock response
        mock_response = create_mock_litellm_response()

        # Capture all warnings during the test
        captured_warnings = []
        original_showwarning = warnings.showwarning

        def capture_warning(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append({
                'message': str(message),
                'category': category.__name__,
                'filename': filename,
                'lineno': lineno
            })
            # Also call original to preserve behavior
            if file is not None:
                original_showwarning(message, category, filename, lineno, file, line)

        with patch.object(warnings, 'showwarning', capture_warning):
            # Mock LiteLLM completion to return our test response
            with patch('litellm.completion', return_value=mock_response):
                # Mock cost tracking callback
                with patch('pdd.llm_invoke._LAST_CALLBACK_DATA', {
                    'cost': 0.001,
                    'input_tokens': 10,
                    'output_tokens': 10
                }):
                    from pdd import cli

                    runner = CliRunner(mix_stderr=False)
                    result = runner.invoke(cli.cli, [
                        "--local",  # Force local execution
                        "generate",
                        "hello.prompt",
                        "--output", "hello.py"
                    ], catch_exceptions=False)

        # Check for Pydantic serialization warnings
        pydantic_warnings = [
            w for w in captured_warnings
            if 'PydanticSerializationUnexpectedValue' in w['message']
        ]

        # Also check stderr output
        stderr_has_pydantic_warning = (
            result.stderr and
            'PydanticSerializationUnexpectedValue' in result.stderr
        )

        # THIS ASSERTION SHOULD FAIL until fix is applied
        assert len(pydantic_warnings) == 0 and not stderr_has_pydantic_warning, (
            f"E2E FAILURE: pdd generate emits Pydantic serialization warnings!\n"
            f"Warnings captured: {pydantic_warnings}\n"
            f"Stderr: {result.stderr[:500] if result.stderr else 'None'}\n\n"
            f"FIX NEEDED: Add warning filter in pdd/llm_invoke.py:\n"
            f"  warnings.filterwarnings(\n"
            f"    'ignore',\n"
            f"    message='.*PydanticSerializationUnexpectedValue.*',\n"
            f"    category=UserWarning\n"
            f"  )"
        )


class TestE2EPydanticWarningsCliTest:
    """E2E tests for Pydantic warning suppression in pdd test command."""

    def test_test_command_does_not_emit_pydantic_warnings(self, tmp_path, monkeypatch):
        """
        E2E TEST: pdd test command should NOT emit Pydantic serialization warnings.
        """
        # Set up test files
        (tmp_path / "module.prompt").write_text("A module that adds two numbers")
        (tmp_path / "module.py").write_text("def add(a, b):\n    return a + b\n")

        monkeypatch.chdir(tmp_path)

        mock_response = create_mock_litellm_response()

        captured_warnings = []
        original_showwarning = warnings.showwarning

        def capture_warning(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append({
                'message': str(message),
                'category': category.__name__,
            })

        with patch.object(warnings, 'showwarning', capture_warning):
            with patch('litellm.completion', return_value=mock_response):
                with patch('pdd.llm_invoke._LAST_CALLBACK_DATA', {
                    'cost': 0.001,
                    'input_tokens': 10,
                    'output_tokens': 10
                }):
                    from pdd import cli

                    runner = CliRunner(mix_stderr=False)
                    result = runner.invoke(cli.cli, [
                        "--local",
                        "test",
                        "module.prompt",
                        "module.py",
                        "--output", "test_module.py"
                    ], catch_exceptions=False)

        pydantic_warnings = [
            w for w in captured_warnings
            if 'PydanticSerializationUnexpectedValue' in w['message']
        ]

        stderr_has_pydantic_warning = (
            result.stderr and
            'PydanticSerializationUnexpectedValue' in result.stderr
        )

        assert len(pydantic_warnings) == 0 and not stderr_has_pydantic_warning, (
            f"E2E FAILURE: pdd test emits Pydantic serialization warnings!\n"
            f"Warnings captured: {pydantic_warnings}\n"
            f"Stderr: {result.stderr[:500] if result.stderr else 'None'}"
        )


class TestE2EPydanticWarningsCliCrash:
    """E2E tests for Pydantic warning suppression in pdd crash command."""

    def test_crash_command_does_not_emit_pydantic_warnings(self, tmp_path, monkeypatch):
        """
        E2E TEST: pdd crash command should NOT emit Pydantic serialization warnings.

        This specifically tests the command mentioned in issue #302.
        """
        # Set up test files mimicking the crash command usage
        (tmp_path / "module.prompt").write_text("A module that greets users")
        (tmp_path / "module.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n")
        (tmp_path / "program.py").write_text("from module import greet\nprint(greet('World'))\n")
        (tmp_path / "error.txt").write_text("NameError: name 'undefined_var' is not defined")

        monkeypatch.chdir(tmp_path)

        mock_response = create_mock_litellm_response()

        captured_warnings = []

        def capture_warning(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append({
                'message': str(message),
                'category': category.__name__,
            })

        with patch.object(warnings, 'showwarning', capture_warning):
            with patch('litellm.completion', return_value=mock_response):
                with patch('pdd.llm_invoke._LAST_CALLBACK_DATA', {
                    'cost': 0.001,
                    'input_tokens': 10,
                    'output_tokens': 10
                }):
                    from pdd import cli

                    runner = CliRunner(mix_stderr=False)
                    result = runner.invoke(cli.cli, [
                        "--local",
                        "crash",
                        "module.prompt",
                        "module.py",
                        "program.py",
                        "error.txt",
                        "--output", "module_fixed.py",
                        "--output-program", "program_fixed.py"
                    ], catch_exceptions=False)

        pydantic_warnings = [
            w for w in captured_warnings
            if 'PydanticSerializationUnexpectedValue' in w['message']
        ]

        stderr_has_pydantic_warning = (
            result.stderr and
            'PydanticSerializationUnexpectedValue' in result.stderr
        )

        assert len(pydantic_warnings) == 0 and not stderr_has_pydantic_warning, (
            f"E2E FAILURE: pdd crash emits Pydantic serialization warnings!\n"
            f"Warnings captured: {pydantic_warnings}\n"
            f"Stderr: {result.stderr[:500] if result.stderr else 'None'}"
        )


class TestE2EPydanticWarningsWithCaching:
    """E2E tests that exercise the LiteLLM caching code path.

    The bug specifically occurs during caching operations where ModelResponse
    objects are serialized. These tests ensure the full caching path is exercised.
    """

    def test_llm_invoke_with_caching_does_not_emit_pydantic_warnings(self, tmp_path, monkeypatch):
        """
        E2E TEST: LiteLLM caching operations should NOT emit Pydantic warnings.

        This test exercises the exact code path that triggers the warning:
        1. LiteLLM returns a ModelResponse
        2. The cache layer attempts to serialize it
        3. Pydantic's model_dump_json() is called
        4. Warning is triggered due to missing attributes

        This test uses actual LiteLLM types but mocks the HTTP call.
        """
        import os

        # Set up cache directory
        cache_dir = tmp_path / "litellm_cache"
        cache_dir.mkdir()
        monkeypatch.setenv("LITELLM_CACHE_PATH", str(cache_dir))

        # Create real LiteLLM response that would trigger the warning
        from litellm.types.utils import ModelResponse, Message, Choices, Usage

        msg = Message(content='{"result": "cached response"}', role="assistant")
        choice = Choices(finish_reason="stop", index=0, message=msg)

        response = ModelResponse(
            id="cache-test-id",
            choices=[choice],
            created=1234567890,
            model="test-model",
            object="chat.completion"
        )

        captured_warnings = []

        def capture_warning(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append({
                'message': str(message),
                'category': category.__name__,
            })

        with patch.object(warnings, 'showwarning', capture_warning):
            # Import llm_invoke after setting up warning capture
            import pdd.llm_invoke
            import importlib
            importlib.reload(pdd.llm_invoke)

            # Now trigger serialization of the response (simulates caching)
            # This is what happens internally in LiteLLM's cache
            try:
                _ = response.model_dump_json()
            except Exception:
                pass  # Ignore errors, we just care about warnings

        pydantic_warnings = [
            w for w in captured_warnings
            if 'PydanticSerializationUnexpectedValue' in w['message']
        ]

        # THIS ASSERTION SHOULD FAIL until fix is applied
        assert len(pydantic_warnings) == 0, (
            f"E2E FAILURE: LiteLLM response serialization emits Pydantic warnings!\n"
            f"This is the root cause of issue #302.\n"
            f"Warnings: {pydantic_warnings}\n\n"
            f"After importing pdd.llm_invoke, Pydantic serialization warnings "
            f"should be suppressed via warnings.filterwarnings()."
        )


class TestE2EWarningSuppressionVerification:
    """Verification tests to confirm warning suppression is properly configured."""

    def test_pdd_llm_invoke_configures_warning_filter_at_import(self):
        """
        E2E TEST: Importing pdd.llm_invoke should configure Pydantic warning filters.

        After the fix, importing the module should add appropriate warning filters
        to suppress PydanticSerializationUnexpectedValue warnings.
        """
        import pdd.llm_invoke
        import importlib

        # Reload to ensure fresh configuration
        importlib.reload(pdd.llm_invoke)

        # Check if any filter suppresses Pydantic serialization warnings
        has_pydantic_filter = False
        for filter_entry in warnings.filters:
            action, message_regex, category, module_regex, lineno = filter_entry

            if action == 'ignore' and category == UserWarning:
                # Check if message pattern matches PydanticSerializationUnexpectedValue
                if message_regex and 'PydanticSerializationUnexpectedValue' in str(message_regex.pattern):
                    has_pydantic_filter = True
                    break
                # Or check module pattern for pydantic
                if module_regex and 'pydantic' in str(module_regex.pattern).lower():
                    has_pydantic_filter = True
                    break

        # THIS ASSERTION SHOULD FAIL until fix is applied
        assert has_pydantic_filter, (
            "E2E FAILURE: pdd.llm_invoke does not configure Pydantic warning filter!\n"
            "Expected a warning filter like:\n"
            "  warnings.filterwarnings(\n"
            "    'ignore',\n"
            "    message='.*PydanticSerializationUnexpectedValue.*',\n"
            "    category=UserWarning\n"
            "  )\n"
            f"Current filters: {[f for f in warnings.filters if f[2] == UserWarning][:5]}"
        )
