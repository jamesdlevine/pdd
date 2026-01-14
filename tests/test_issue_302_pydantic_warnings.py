# Test for GitHub Issue #302: Pydantic serialization warnings from LiteLLM
# https://github.com/promptdriven/pdd/issues/302
#
# This test verifies that Pydantic serialization warnings from LiteLLM's
# response models (Message, Choices, etc.) are properly suppressed.
#
# Root Cause: LiteLLM's response models delete optional attributes after
# initialization to match OpenAI's API spec. When Pydantic serializes these
# objects (during caching operations), it expects all defined fields but
# finds fewer actual attributes, triggering UserWarning with
# PydanticSerializationUnexpectedValue.
#
# This is an upstream LiteLLM bug (BerriAI/litellm#11759), but PDD should
# suppress these cosmetic warnings to prevent noisy output for users.

import pytest
import warnings
from unittest.mock import patch, MagicMock


class TestPydanticSerializationWarnings:
    """Tests for Issue #302: Pydantic warnings from LiteLLM serialization."""

    def test_litellm_model_response_serialization_emits_warning(self):
        """Verify that LiteLLM ModelResponse serialization triggers Pydantic warning.

        This test documents the upstream LiteLLM behavior that causes the warnings.
        It should always pass - it's a characterization test of the underlying issue.
        """
        from litellm.types.utils import ModelResponse, Message, Choices

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create a ModelResponse with Message - same pattern as LiteLLM uses
            msg = Message(content="test response", role="assistant")
            choice = Choices(finish_reason="stop", index=0, message=msg)
            resp = ModelResponse(
                id="test-123",
                choices=[choice],
                created=1234567890,
                model="test-model",
                object="chat.completion"
            )

            # Serialization triggers the warning
            _ = resp.model_dump_json()

            # Verify warning is emitted (documenting upstream behavior)
            pydantic_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "PydanticSerializationUnexpectedValue" in str(warning.message)
            ]

            # This assertion documents the current LiteLLM behavior
            assert len(pydantic_warnings) >= 1, (
                "Expected LiteLLM ModelResponse serialization to emit "
                "Pydantic warnings, but none were captured. "
                "This may indicate LiteLLM has fixed the upstream issue."
            )

    def test_pdd_suppresses_litellm_pydantic_warnings(self):
        """Verify PDD suppresses Pydantic serialization warnings from LiteLLM.

        Issue #302: When using PDD with LLM providers (especially with caching),
        Pydantic serialization warnings are emitted, causing noisy output.

        PDD should suppress these warnings since they are:
        1. Cosmetic only - they don't affect functionality
        2. From an upstream dependency (LiteLLM)
        3. Distracting for users

        EXPECTED BEHAVIOR AFTER FIX:
        - Importing pdd.llm_invoke should configure warning filters
        - LiteLLM response serialization should NOT emit warnings to users
        """
        from litellm.types.utils import ModelResponse, Message, Choices

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import llm_invoke - this should set up warning filters
            import pdd.llm_invoke

            # Force reload to ensure warning configuration is applied fresh
            import importlib
            importlib.reload(pdd.llm_invoke)

            # Now create and serialize a ModelResponse
            msg = Message(content="test response", role="assistant")
            choice = Choices(finish_reason="stop", index=0, message=msg)
            resp = ModelResponse(
                id="test-123",
                choices=[choice],
                created=1234567890,
                model="test-model",
                object="chat.completion"
            )

            # Serialization would normally trigger the warning
            _ = resp.model_dump_json()

            # After the fix, PDD should suppress these warnings
            pydantic_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "PydanticSerializationUnexpectedValue" in str(warning.message)
            ]

            # THIS IS THE FAILING ASSERTION - currently warnings ARE emitted
            assert len(pydantic_warnings) == 0, (
                f"PDD should suppress Pydantic serialization warnings from LiteLLM. "
                f"Found {len(pydantic_warnings)} warning(s): "
                f"{[str(w.message)[:100] for w in pydantic_warnings]}"
            )

    def test_llm_invoke_does_not_emit_pydantic_warnings_during_call(self):
        """Verify llm_invoke() doesn't emit Pydantic warnings during LLM calls.

        This tests the full code path through llm_invoke() to ensure warnings
        are suppressed during actual usage, not just at module import time.
        """
        import os
        import json
        from pdd.llm_invoke import llm_invoke

        # Create a mock response that would trigger warnings if serialized
        def create_mock_response():
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_usage = MagicMock()

            mock_message.content = json.dumps({"result": "test"})
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 10
            mock_usage.total_tokens = 20
            mock_response.usage = mock_usage
            mock_response.model = "test-model"
            mock_response._hidden_params = {}
            return mock_response

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock all the internals to focus on warning behavior
            with patch('pdd.llm_invoke._load_model_data') as mock_load:
                import pandas as pd
                mock_load.return_value = pd.DataFrame([{
                    'provider': 'OpenAI',
                    'model': 'gpt-test',
                    'input': 0.01,
                    'output': 0.02,
                    'coding_arena_elo': 1500,
                    'structured_output': True,
                    'base_url': '',
                    'api_key': 'OPENAI_API_KEY',
                    'max_tokens': '',
                    'max_completion_tokens': '',
                    'reasoning_type': 'none',
                    'max_reasoning_tokens': 0,
                    'avg_cost': 0.015,
                }])

                with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake_key'}):
                    with patch('pdd.llm_invoke.litellm.completion') as mock_completion:
                        mock_completion.return_value = create_mock_response()

                        with patch('pdd.llm_invoke._LAST_CALLBACK_DATA', {
                            'cost': 0.001,
                            'input_tokens': 10,
                            'output_tokens': 10
                        }):
                            try:
                                result = llm_invoke(
                                    prompt="Test prompt",
                                    input_json={},
                                    strength=0.5,
                                    time=0.5,
                                    verbose=False,
                                )
                            except Exception:
                                # We don't care if the call succeeds, just checking warnings
                                pass

            # Check for Pydantic serialization warnings
            pydantic_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "PydanticSerializationUnexpectedValue" in str(warning.message)
            ]

            assert len(pydantic_warnings) == 0, (
                f"llm_invoke should not emit Pydantic serialization warnings. "
                f"Found {len(pydantic_warnings)} warning(s)"
            )


# Additional test to verify the fix approach
class TestWarningSuppressionConfiguration:
    """Tests for warning suppression configuration in llm_invoke."""

    def test_warning_filter_is_configured_for_pydantic_serialization(self):
        """Verify that llm_invoke configures appropriate warning filters.

        The fix should add a warning filter like:
        warnings.filterwarnings(
            'ignore',
            message='.*PydanticSerializationUnexpectedValue.*',
            category=UserWarning,
            module='pydantic.*'
        )
        """
        import pdd.llm_invoke
        import importlib

        # Reload to ensure fresh configuration
        importlib.reload(pdd.llm_invoke)

        # Check if appropriate filter is in place
        # Note: This is an implementation-specific check - the test above
        # (test_pdd_suppresses_litellm_pydantic_warnings) is the primary test

        pydantic_filters = [
            f for f in warnings.filters
            if (
                f[0] == 'ignore'
                and f[2] == UserWarning
                and (
                    'pydantic' in str(f[3]).lower() if f[3] else False
                    or 'PydanticSerializationUnexpectedValue' in str(f[1]) if f[1] else False
                )
            )
        ]

        # This test documents whether a filter is configured
        # It's expected to fail until the fix is implemented
        assert len(pydantic_filters) >= 1, (
            "Expected warning filter for Pydantic serialization warnings. "
            "The fix should add: warnings.filterwarnings('ignore', "
            "message='.*PydanticSerializationUnexpectedValue.*', "
            "category=UserWarning)"
        )
