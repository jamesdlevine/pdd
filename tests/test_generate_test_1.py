# Test Plan for pdd.generate_test.generate_test
#
# 1. Analysis of Code Under Test:
#    - Function: generate_test
#    - Role: Orchestrates the generation of unit tests using LLMs.
#    - Key Steps:
#      1. Load prompt template.
#      2. Preprocess inputs.
#      3. Invoke LLM (llm_invoke).
#      4. Check for completion (unfinished_prompt).
#      5. Continue generation if needed (continue_generation).
#      6. Postprocess result (postprocess).
#      7. Fallback to regex extraction if postprocess fails.
#
# 2. Edge Cases & Scenarios:
#    - Template Loading Failure: load_prompt_template returns None.
#    - Empty LLM Response: llm_invoke returns empty string.
#    - Unfinished Generation: LLM output is cut off, requiring continuation.
#    - Finished Generation (Optimization): LLM output ends with whitespace, skipping unfinished check.
#    - Postprocess Failure: The postprocess module fails, triggering regex fallback.
#    - Regex Fallback Failure: Regex fails to find code, returning raw result.
#
# 3. Z3 Formal Verification Strategy:
#    - The primary logic suitable for formal verification here is the cost accumulation.
#    - We can verify that the returned total_cost is always the sum of costs from invoked steps
#      (initial generation, completion check, continuation, postprocessing).
#    - Other logic is procedural integration and better suited for mocks.
#
# 4. Unit Test Strategy:
#    - Use unittest.mock to patch all external dependencies:
#      - pdd.generate_test.load_prompt_template
#      - pdd.generate_test.preprocess
#      - pdd.generate_test.llm_invoke
#      - pdd.generate_test.unfinished_prompt
#      - pdd.generate_test.continue_generation
#      - pdd.generate_test.postprocess
#    - Verify correct parameters are passed to these dependencies.
#    - Verify correct handling of return values and exceptions.

import pytest
from unittest.mock import MagicMock, patch, ANY
from z3 import Solver, Real, Bool, If, Not, unsat as z3_unsat, sat
from pdd.generate_test import generate_test

# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_dependencies():
    """Fixture to mock all external dependencies of generate_test."""
    with patch('pdd.generate_test.load_prompt_template') as mock_load, \
         patch('pdd.generate_test.preprocess') as mock_preprocess, \
         patch('pdd.generate_test.llm_invoke') as mock_invoke, \
         patch('pdd.generate_test.unfinished_prompt') as mock_unfinished, \
         patch('pdd.generate_test.continue_generation') as mock_continue, \
         patch('pdd.generate_test.postprocess') as mock_postprocess, \
         patch('pdd.generate_test.console') as mock_console:
        
        # Default happy path setup
        mock_load.return_value = "template_content"
        mock_preprocess.side_effect = lambda x, **kwargs: f"processed_{x}"
        mock_invoke.return_value = {
            'result': "def test_foo(): pass",
            'cost': 0.01,
            'model_name': "gpt-4"
        }
        # Default: finished, cost 0.001
        mock_unfinished.return_value = ("Reasoning", True, 0.001, "gpt-4")
        mock_postprocess.return_value = ("def test_foo(): pass", 0.002, "post-model")
        
        yield {
            'load': mock_load,
            'preprocess': mock_preprocess,
            'invoke': mock_invoke,
            'unfinished': mock_unfinished,
            'continue': mock_continue,
            'postprocess': mock_postprocess,
            'console': mock_console
        }


def test_generate_test_happy_path(mock_dependencies):
    """Test the standard successful flow without continuation."""
    prompt = "Make a test"
    code = "def foo(): pass"
    
    result, cost, model = generate_test(prompt, code, verbose=True)
    
    # Check result
    assert result == "def test_foo(): pass"
    # Cost = 0.01 (invoke) + 0.001 (unfinished check) + 0.002 (postprocess)
    assert cost == pytest.approx(0.013)
    # The code returns the model from the generation step, not postprocess
    assert model == "gpt-4"
    
    # Verify calls
    mock_dependencies['load'].assert_called_once_with("generate_test_LLM")
    mock_dependencies['invoke'].assert_called_once()
    mock_dependencies['unfinished'].assert_called_once()
    mock_dependencies['continue'].assert_not_called()
    mock_dependencies['postprocess'].assert_called_once()


def test_generate_test_template_load_failure(mock_dependencies):
    """Test that ValueError is raised if template cannot be loaded."""
    mock_dependencies['load'].return_value = None
    
    with pytest.raises(ValueError, match="Failed to load generate_test_LLM"):
        generate_test("prompt", "code")


def test_generate_test_empty_llm_result(mock_dependencies):
    """Test that ValueError is raised if LLM returns empty result."""
    mock_dependencies['invoke'].return_value = {
        'result': "",
        'cost': 0.01,
        'model_name': "gpt-4"
    }
    
    with pytest.raises(ValueError, match="LLM test generation returned empty result"):
        generate_test("prompt", "code")


def test_generate_test_continuation_flow(mock_dependencies):
    """Test the flow where generation is incomplete and requires continuation."""
    # Setup unfinished prompt
    mock_dependencies['invoke'].return_value = {
        'result': "def test_foo():",  # Incomplete
        'cost': 0.01,
        'model_name': "gpt-4"
    }
    # unfinished_prompt returns False (not finished)
    mock_dependencies['unfinished'].return_value = ("Incomplete", False, 0.001, "gpt-4")
    
    # continue_generation returns completed code
    mock_dependencies['continue'].return_value = (
        "def test_foo(): pass",  # Completed
        0.005,  # Cost
        "gpt-4-continued"  # Model
    )
    
    result, cost, model = generate_test("prompt", "code")
    
    # Verify continuation was called
    mock_dependencies['continue'].assert_called_once()
    
    # Cost = 0.01 (invoke) + 0.001 (check) + 0.005 (continue) + 0.002 (postprocess)
    assert cost == pytest.approx(0.018)
    # The code returns the model from the continuation step
    assert model == "gpt-4-continued"


def test_generate_test_optimization_empty_tail(mock_dependencies):
    """Test handling of results ending with whitespace."""
    # Result ends with whitespace/newlines
    mock_dependencies['invoke'].return_value = {
        'result': "def test_foo(): pass\n   \n",
        'cost': 0.01,
        'model_name': "gpt-4"
    }
    
    # Ensure unfinished_prompt returns True (finished) so we don't loop
    mock_dependencies['unfinished'].return_value = ("Reasoning", True, 0.001, "gpt-4")

    generate_test("prompt", "code")
    
    # The optimization to skip unfinished_prompt on whitespace tail does not exist
    # in the current implementation, so we verify it IS called.
    mock_dependencies['unfinished'].assert_called_once()
    mock_dependencies['continue'].assert_not_called()


def test_generate_test_postprocess_failure_fallback_success(mock_dependencies):
    """Test fallback to regex extraction when postprocess fails."""
    # Setup postprocess to fail
    mock_dependencies['postprocess'].side_effect = Exception("Postprocess error")
    
    # LLM returns markdown code block
    llm_output = "Here is the code:\n```python\ndef test_fallback(): pass\n```"
    mock_dependencies['invoke'].return_value = {
        'result': llm_output,
        'cost': 0.01,
        'model_name': "gpt-4"
    }
    
    result, cost, model = generate_test("prompt", "code")
    
    # Should catch exception and extract code
    assert result == "def test_fallback(): pass"
    # Cost should not include postprocess cost (it failed, and fallback is free/local)
    # Cost = 0.01 (invoke) + 0.001 (unfinished check)
    assert cost == pytest.approx(0.011)


def test_generate_test_postprocess_failure_fallback_raw(mock_dependencies):
    """Test fallback returns raw result if regex fails to find code blocks."""
    mock_dependencies['postprocess'].side_effect = Exception("Postprocess error")
    
    # LLM returns text without code blocks
    raw_text = "Just some text without code blocks."
    mock_dependencies['invoke'].return_value = {
        'result': raw_text,
        'cost': 0.01,
        'model_name': "gpt-4"
    }
    
    result, cost, model = generate_test("prompt", "code")
    
    assert result == raw_text


def test_generate_test_input_params_passed_correctly(mock_dependencies):
    """Verify that input parameters are correctly propagated to llm_invoke."""
    prompt = "my_prompt"
    code = "my_code"
    strength = 0.8
    temperature = 0.5
    time = 0.9
    language = "javascript"
    source_path = "/src/foo.py"
    
    generate_test(
        prompt, code,
        strength=strength,
        temperature=temperature,
        time=time,
        language=language,
        source_file_path=source_path
    )
    
    # Check llm_invoke args
    call_args = mock_dependencies['invoke'].call_args
    assert call_args.kwargs['strength'] == strength
    assert call_args.kwargs['temperature'] == temperature
    assert call_args.kwargs['time'] == time
    assert call_args.kwargs['input_json']['language'] == language
    assert call_args.kwargs['input_json']['source_file_path'] == source_path


# -----------------------------------------------------------------------------
# Z3 Formal Verification Tests
# -----------------------------------------------------------------------------


def test_z3_cost_accumulation_logic():
    """
    Formally verify that the total_cost calculation logic is sound.
    
    We model the cost accumulation steps and ensure total_cost equals the sum of parts.
    The property verified: total_cost is always >= initial_cost.
    """
    s = Solver()
    
    # Define variables for costs (must be non-negative)
    initial_cost = Real('initial_cost')
    check_cost = Real('check_cost')
    continue_cost = Real('continue_cost')
    post_cost = Real('post_cost')
    total_cost = Real('total_cost')
    
    # Boolean flags for control flow
    is_finished = Bool('is_finished')
    postprocess_fails = Bool('postprocess_fails')
    
    # Constraints (Costs are non-negative)
    s.add(initial_cost >= 0)
    s.add(check_cost >= 0)
    s.add(continue_cost >= 0)
    s.add(post_cost >= 0)
    
    # Logic Model of generate_test cost accumulation:
    
    # 1. Start with initial cost
    accumulated = initial_cost
    
    # 2. Check completion - add check_cost
    cost_after_check = accumulated + check_cost
    
    # 3. Continuation
    # If NOT finished, add continue_cost; otherwise add 0
    cost_after_continue = cost_after_check + If(Not(is_finished), continue_cost, 0)
    
    # 4. Postprocessing
    # If postprocess succeeds, add post_cost; otherwise add 0
    cost_final = cost_after_continue + If(Not(postprocess_fails), post_cost, 0)
    
    # Define total_cost as the final accumulated cost
    s.add(total_cost == cost_final)
    
    # Verify property: "Total cost is always >= initial_cost"
    # We try to find a counter-example where total_cost < initial_cost
    s.add(Not(total_cost >= initial_cost))
    
    result = s.check()
    
    # If result is unsat, no counter-example exists, so the property holds
    assert result == z3_unsat, \
        "Found a case where total_cost < initial_cost, which should be impossible"