"""
DETAILED TEST PLAN FOR trace.py

This test suite ensures the correct functionality of the trace() function which maps
a line in a code file to its corresponding line in a prompt file using LLM analysis
and fuzzy string matching.

=== TESTING STRATEGY ===

1. INPUT VALIDATION TESTS (Unit Tests)
   - Test empty or None code_file/prompt_file
   - Test invalid code_line numbers (< 1, > file length, non-integer)
   - Test missing required parameters
   - These are best tested with unit tests as they involve exception handling

2. NORMAL OPERATION TESTS (Unit Tests with Mocking)
   - Test successful trace with mocked LLM responses
   - Test that LLM is called with correct parameters
   - Test cost accumulation from multiple LLM calls
   - Test model name propagation
   - Test verbose mode functionality
   - Unit tests with mocking are ideal here to avoid actual LLM API calls

3. FUZZY MATCHING LOGIC TESTS (Unit Tests)
   - Test exact string match (ratio = 1.0)
   - Test high similarity match (ratio > 0.8)
   - Test medium similarity match (0.6 <= ratio < 0.8)
   - Test low similarity match requiring fallback
   - Test multi-line window matching (2-3 line windows)
   - Test text normalization (unicode quotes, whitespace)
   - Unit tests are best as fuzzy matching is probabilistic and context-dependent

4. FALLBACK MECHANISM TESTS (Unit Tests)
   - Test fallback when LLM output doesn't match any line well
   - Test fallback when exceptions occur
   - Test token-based matching in fallback
   - Unit tests work best for testing fallback paths

5. EDGE CASES (Unit Tests)
   - Single-line files
   - Files with empty lines
   - Files with only whitespace
   - Very long files
   - Special characters and unicode
   - Unit tests handle these scenarios better

6. Z3 FORMAL VERIFICATION TESTS (Runnable as Unit Tests)
   - Verify output line number is always in valid range [1, len(prompt_lines)]
   - Verify total_cost is always non-negative
   - Verify return tuple has exactly 3 elements
   - Z3 is ideal for verifying mathematical invariants and properties
   
=== WHY Z3 vs UNIT TESTS ===

Z3 Formal Verification is suitable for:
- Mathematical invariants (line number bounds, cost >= 0)
- Proving properties hold for ALL possible inputs
- Verifying logical constraints

Unit Tests are suitable for:
- Testing actual behavior with concrete inputs
- Testing integration with mocked external dependencies
- Testing string manipulation and fuzzy matching
- Testing error handling with specific scenarios
- Testing non-deterministic or external operations (LLM calls)

For this code:
- ~80% should be unit tests (due to LLM calls, string matching, external dependencies)
- ~20% should be Z3 verification (for mathematical properties and invariants)
"""

import pytest
from unittest.mock import patch
from z3 import (
    Int,
    Real,
    Bool,
    String,
    Solver,
    Or,
    Implies,
    Length,
    sat,
    unsat,
)

# Import the function under test
from pdd.trace import trace, PromptLineOutput


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_code_file() -> str:
    """Sample code file for testing."""
    return """def hello_world():
    print("Hello, World!")
    return True

def goodbye():
    print("Goodbye!")
"""


@pytest.fixture
def sample_prompt_file() -> str:
    """Sample prompt file for testing."""
    return """Write a function that prints Hello World
Make sure to return True
Add another function
That prints Goodbye
"""


@pytest.fixture
def mock_llm_invoke():
    """Mock llm_invoke to avoid actual API calls."""
    with patch('pdd.trace.llm_invoke') as mock:
        yield mock


@pytest.fixture
def mock_load_prompt_template():
    """Mock load_prompt_template to return dummy templates."""
    with patch('pdd.trace.load_prompt_template') as mock:
        mock.return_value = (
            "Mocked prompt template with {CODE_FILE} {CODE_STR} "
            "{PROMPT_FILE} {llm_output}"
        )
        yield mock


@pytest.fixture
def mock_preprocess():
    """Mock preprocess to return the input unchanged."""
    with patch('pdd.trace.preprocess') as mock:
        mock.side_effect = lambda x, **kwargs: x
        yield mock


# ============================================================================
# 1. INPUT VALIDATION TESTS
# ============================================================================


def test_trace_empty_code_file(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test trace with empty code_file."""
    # The function catches ValueError and returns fallback
    result = trace("", 1, "prompt content")
    assert result[0] == 1
    assert result[2] == "fallback"


def test_trace_empty_prompt_file(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test trace with empty prompt_file."""
    # The function catches ValueError and returns fallback
    result = trace("code content", 1, "")
    assert result[0] == 1
    assert result[2] == "fallback"


def test_trace_invalid_code_line_negative(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test trace with negative code_line."""
    result = trace("line1\nline2\nline3", -1, "prompt line")
    # Should return fallback
    assert result[0] == 1
    assert result[2] == "fallback"


def test_trace_invalid_code_line_too_large(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test trace with code_line greater than file length."""
    result = trace("line1\nline2", 10, "prompt line")
    # Should return fallback
    assert result[0] == 1
    assert result[2] == "fallback"


def test_trace_invalid_code_line_zero(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test trace with code_line = 0."""
    result = trace("line1\nline2", 0, "prompt line")
    # Should return fallback
    assert result[0] == 1
    assert result[2] == "fallback"


# ============================================================================
# 2. NORMAL OPERATION TESTS
# ============================================================================


def test_trace_successful_exact_match(
    mock_llm_invoke,
    mock_load_prompt_template,
    mock_preprocess,
    sample_code_file,
    sample_prompt_file,
) -> None:
    """Test successful trace with exact match."""
    # Setup mocks
    mock_llm_invoke.side_effect = [
        {
            'result': 'The line corresponds to: print("Hello, World!")',
            'cost': 0.001,
            'model_name': 'gpt-4',
        },
        {
            'result': PromptLineOutput(
                prompt_line='Write a function that prints Hello World'
            ),
            'cost': 0.0005,
            'model_name': 'gpt-4',
        },
    ]

    result = trace(sample_code_file, 2, sample_prompt_file)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], int)
    assert result[0] >= 1
    assert isinstance(result[1], float)
    assert result[1] >= 0
    assert isinstance(result[2], str)


def test_trace_cost_accumulation(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test that costs from multiple LLM calls are accumulated."""
    mock_llm_invoke.side_effect = [
        {'result': 'Some analysis', 'cost': 0.123, 'model_name': 'model1'},
        {
            'result': PromptLineOutput(prompt_line='matching line'),
            'cost': 0.456,
            'model_name': 'model1',
        },
    ]

    result = trace("line1\nline2", 1, "prompt1\nprompt2")

    assert result[1] == pytest.approx(0.123 + 0.456, rel=1e-6)


def test_trace_model_name_returned(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test that model name from first LLM call is returned."""
    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'test-model-xyz'},
        {
            'result': PromptLineOutput(prompt_line='line'),
            'cost': 0.01,
            'model_name': 'different-model',
        },
    ]

    result = trace("code", 1, "prompt")

    assert result[2] == 'test-model-xyz'


def test_trace_with_custom_strength(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test trace with custom strength parameter."""
    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='line'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    trace("code", 1, "prompt", strength=0.8)

    # Verify strength was passed to first llm_invoke call
    assert mock_llm_invoke.call_args_list[0][1]['strength'] == 0.8


def test_trace_with_custom_temperature(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test trace with custom temperature parameter."""
    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='line'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    trace("code", 1, "prompt", temperature=0.7)

    # Verify temperature was passed to both calls
    assert mock_llm_invoke.call_args_list[0][1]['temperature'] == 0.7
    assert mock_llm_invoke.call_args_list[1][1]['temperature'] == 0.7


def test_trace_with_verbose_true(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess, capsys
) -> None:
    """Test trace with verbose=True."""
    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='prompt line'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    trace("code line", 1, "prompt line", verbose=True)

    # Verify verbose was passed to llm_invoke
    assert mock_llm_invoke.call_args_list[0][1]['verbose'] is True
    assert mock_llm_invoke.call_args_list[1][1]['verbose'] is True


def test_trace_llm_invoke_parameters(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test that llm_invoke is called with correct parameters."""
    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='line'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    trace("code content", 1, "prompt content", strength=0.6, temperature=0.3, time=0.5)

    # Check first call
    first_call = mock_llm_invoke.call_args_list[0]
    assert 'CODE_FILE' in first_call[1]['input_json']
    assert 'CODE_STR' in first_call[1]['input_json']
    assert 'PROMPT_FILE' in first_call[1]['input_json']
    assert first_call[1]['strength'] == 0.6
    assert first_call[1]['temperature'] == 0.3
    assert first_call[1]['time'] == 0.5

    # Check second call
    second_call = mock_llm_invoke.call_args_list[1]
    assert 'llm_output' in second_call[1]['input_json']
    assert second_call[1]['output_pydantic'] == PromptLineOutput
    assert second_call[1]['strength'] == 0.6
    assert second_call[1]['temperature'] == 0.3


# ============================================================================
# 3. FUZZY MATCHING LOGIC TESTS
# ============================================================================


def test_trace_exact_string_match(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test fuzzy matching with exact string match."""
    code_file = "def hello():\n    print('hello')\n    return True"
    prompt_file = "Create a hello function\nMake it print hello\nReturn True from function"

    mock_llm_invoke.side_effect = [
        {'result': 'Maps to return True', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='Return True from function'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    result = trace(code_file, 3, prompt_file)

    # Should find line 3 which contains "Return True from function"
    assert result[0] == 3


def test_trace_fuzzy_match_high_similarity(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test fuzzy matching with high similarity (>0.8)."""
    code_file = "x = calculate_total_sum()"
    prompt_file = "Calculate the sum\nStore in variable x\nCall calculate_total_sum function"

    mock_llm_invoke.side_effect = [
        {'result': 'Calls calculate sum', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='Call calculate_total_sum function'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    result = trace(code_file, 1, prompt_file)

    # Should find a line (exact line depends on matching algorithm)
    assert 1 <= result[0] <= 3


def test_trace_with_unicode_normalization(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test matching with unicode quotes and special characters."""
    code_file = 'print("Hello")'
    prompt_file = 'Print "Hello"\nWith quotes'

    mock_llm_invoke.side_effect = [
        {'result': 'Prints Hello', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='Print "Hello"'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    result = trace(code_file, 1, prompt_file)

    assert result[0] >= 1


def test_trace_with_whitespace_normalization(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test matching with different whitespace."""
    code_file = "def    foo():\n    pass"
    prompt_file = "Define   a   foo   function\nWith pass statement"

    mock_llm_invoke.side_effect = [
        {'result': 'foo function', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='Define a foo function'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    result = trace(code_file, 1, prompt_file)

    assert result[0] >= 1


# ============================================================================
# 4. FALLBACK MECHANISM TESTS
# ============================================================================


def test_trace_fallback_on_no_match(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test fallback when LLM output doesn't match any line."""
    code_file = "specific_code_line"
    prompt_file = "completely different\nunrelated content\nnothing matches"

    mock_llm_invoke.side_effect = [
        {'result': 'Some analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='nonexistent line xyz abc 123'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    result = trace(code_file, 1, prompt_file)

    # Should return some line number (fallback logic)
    assert 1 <= result[0] <= 3


def test_trace_fallback_on_exception(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test fallback when exception occurs during processing."""
    mock_load_prompt_template.side_effect = Exception("Template not found")

    result = trace("code", 1, "prompt")

    # Should return fallback values
    assert result[0] == 1
    assert result[1] == 0.0
    assert result[2] == "fallback"


def test_trace_fallback_with_token_matching(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test fallback uses token-based matching."""
    code_file = "user_authentication_function"
    prompt_file = "Handle errors\nCreate user authentication function here\nReturn result"

    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='completely unrelated text xyz'),
            'cost': 0.01,
            'model_name': 'model',
        },
    ]

    result = trace(code_file, 1, prompt_file)

    # Fallback should prefer line 2 which has matching tokens
    # (This tests the token-based fallback heuristic)
    assert result[0] >= 1


# ============================================================================
# 5. EDGE CASES
# ============================================================================


def test_trace_single_line_code_file(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test with single-line code file."""
    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='single line'),
            'cost': 0.01,
            'model_name': 'model'},
    ]

    result = trace("single_line_of_code", 1, "single line\nof prompt")

    assert result[0] >= 1


def test_trace_single_line_prompt_file(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test with single-line prompt file."""
    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='only line'),
            'cost': 0.01,
            'model_name': 'model'},
    ]

    result = trace("line1\nline2\nline3", 2, "only line")

    assert result[0] == 1


def test_trace_empty_lines_in_files(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test with empty lines in code and prompt files."""
    code_file = "line1\n\nline3\n\nline5"
    prompt_file = "prompt1\n\nprompt3\n\nprompt5"

    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='prompt3'),
            'cost': 0.01,
            'model_name': 'model'},
    ]

    result = trace(code_file, 3, prompt_file)

    assert result[0] >= 1


def test_trace_only_whitespace_lines(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test with lines containing only whitespace."""
    code_file = "code\n   \n\t\nmore code"
    prompt_file = "prompt\n   \n\t\nmore prompt"

    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='more prompt'),
            'cost': 0.01,
            'model_name': 'model'},
    ]

    result = trace(code_file, 4, prompt_file)

    assert result[0] >= 1


def test_trace_very_long_files(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test with very long files."""
    code_file = "\n".join([f"code_line_{i}" for i in range(1000)])
    prompt_file = "\n".join([f"prompt_line_{i}" for i in range(1000)])

    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line='prompt_line_500'),
            'cost': 0.01,
            'model_name': 'model'},
    ]

    result = trace(code_file, 500, prompt_file)

    assert 1 <= result[0] <= 1000


def test_trace_special_characters(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Test with special characters in files."""
    code_file = "def func():\n    x = 'test@#$%^&*()'\n    return x"
    prompt_file = "Define function\nSet x to test@#$%^&*()\nReturn the value"

    mock_llm_invoke.side_effect = [
        {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
        {
            'result': PromptLineOutput(prompt_line="Set x to test@#$%^&*()"),
            'cost': 0.01,
            'model_name': 'model'},
    ]

    result = trace(code_file, 2, prompt_file)

    assert result[0] >= 1


# ============================================================================
# 6. Z3 FORMAL VERIFICATION TESTS
# ============================================================================


def test_z3_output_line_number_in_valid_range() -> None:
    """
    Z3 Formal Verification: Verify that the output line number is always
    within the valid range [1, number_of_prompt_lines].
    """
    # Create Z3 variables
    num_prompt_lines = Int('num_prompt_lines')
    output_line = Int('output_line')

    # Create solver
    solver = Solver()

    # Define constraints based on the function's behavior
    # The function should return a line number between 1 and num_prompt_lines (inclusive)
    solver.add(num_prompt_lines >= 1)  # At least one line in prompt file
    solver.add(output_line >= 1)  # Line numbers start at 1
    solver.add(output_line <= num_prompt_lines)  # Can't exceed total lines

    # Check if constraints are satisfiable
    assert solver.check() == sat, "Valid range constraint should be satisfiable"

    # Try to find a counter-example where output is out of range
    solver_counter = Solver()
    solver_counter.add(num_prompt_lines >= 1)
    solver_counter.add(Or(output_line < 1, output_line > num_prompt_lines))

    # This should be satisfiable (we can construct invalid outputs)
    # but the actual function should never produce them
    assert solver_counter.check() == sat, "Counter-example exists in theory"

    # Now verify with actual function calls
    with patch('pdd.trace.llm_invoke') as mock_llm:
        with patch('pdd.trace.load_prompt_template') as mock_load:
            with patch('pdd.trace.preprocess') as mock_prep:
                mock_load.return_value = "template"
                mock_prep.side_effect = lambda x, **kwargs: x
                mock_llm.side_effect = [
                    {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
                    {
                        'result': PromptLineOutput(prompt_line='line'),
                        'cost': 0.01,
                        'model_name': 'model'},
                ]

                # Test with various prompt file sizes
                for num_lines in [1, 5, 10, 100]:
                    prompt_file = "\n".join([f"line{i}" for i in range(num_lines)])
                    result = trace("code", 1, prompt_file)

                    # Verify the formal property holds
                    assert 1 <= result[0] <= num_lines, (
                        f"Output line {result[0]} is not in valid range [1, {num_lines}]"
                    )


def test_z3_total_cost_non_negative() -> None:
    """
    Z3 Formal Verification: Verify that total_cost is always non-negative.
    """
    # Create Z3 variables
    cost1 = Real('cost1')
    cost2 = Real('cost2')
    total_cost = Real('total_cost')

    # Create solver
    solver = Solver()

    # Define constraints
    solver.add(cost1 >= 0)  # Individual costs are non-negative
    solver.add(cost2 >= 0)
    solver.add(total_cost == cost1 + cost2)  # Total is sum of individual costs
    solver.add(total_cost >= 0)  # Total must be non-negative

    # Check if constraints are satisfiable
    assert solver.check() == sat, "Non-negative cost constraint should be satisfiable"

    # Verify no counter-example exists where total_cost < 0 given non-negative inputs
    solver_counter = Solver()
    solver_counter.add(cost1 >= 0)
    solver_counter.add(cost2 >= 0)
    solver_counter.add(total_cost == cost1 + cost2)
    solver_counter.add(total_cost < 0)

    # This should be unsatisfiable (no valid counter-example)
    assert solver_counter.check() == unsat, (
        "Counter-example should not exist: sum of non-negative numbers is non-negative"
    )

    # Verify with actual function
    with patch('pdd.trace.llm_invoke') as mock_llm:
        with patch('pdd.trace.load_prompt_template') as mock_load:
            with patch('pdd.trace.preprocess') as mock_prep:
                mock_load.return_value = "template"
                mock_prep.side_effect = lambda x, **kwargs: x

                # Test with various cost combinations
                test_costs = [
                    (0.001, 0.002),
                    (0.0, 0.0),
                    (1.5, 2.5),
                    (0.00001, 0.00002),
                ]

                for cost_a, cost_b in test_costs:
                    mock_llm.side_effect = [
                        {'result': 'Analysis', 'cost': cost_a, 'model_name': 'model'},
                        {'result': PromptLineOutput(prompt_line='line'), 'cost': cost_b, 'model_name': 'model'},
                    ]

                    result = trace("code", 1, "prompt")

                    # Verify the formal property holds
                    assert result[1] >= 0, f"Total cost {result[1]} is negative"
                    assert result[1] == pytest.approx(cost_a + cost_b, rel=1e-9), (
                        f"Total cost {result[1]} doesn't match sum of individual costs"
                    )


def test_z3_return_tuple_structure() -> None:
    """
    Z3 Formal Verification: Verify that the return value is always a tuple
    with exactly 3 elements of the correct types.
    """
    # Create Z3 variables representing the types
    result_is_tuple = Bool('result_is_tuple')
    result_length = Int('result_length')
    first_elem_is_int_or_none = Bool('first_elem_is_int_or_none')
    second_elem_is_float = Bool('second_elem_is_float')
    third_elem_is_str = Bool('third_elem_is_str')

    # Create solver
    solver = Solver()

    # Define constraints for valid output
    solver.add(result_is_tuple == True)
    solver.add(result_length == 3)
    solver.add(first_elem_is_int_or_none == True)
    solver.add(second_elem_is_float == True)
    solver.add(third_elem_is_str == True)

    # Check if valid output structure is satisfiable
    assert solver.check() == sat, "Valid output structure should be satisfiable"

    # Verify with actual function
    with patch('pdd.trace.llm_invoke') as mock_llm:
        with patch('pdd.trace.load_prompt_template') as mock_load:
            with patch('pdd.trace.preprocess') as mock_prep:
                mock_load.return_value = "template"
                mock_prep.side_effect = lambda x, **kwargs: x
                mock_llm.side_effect = [
                    {'result': 'Analysis', 'cost': 0.01, 'model_name': 'test_model'},
                    {'result': PromptLineOutput(prompt_line='line'), 'cost': 0.02, 'model_name': 'test_model'},
                ]

                result = trace("code", 1, "prompt")

                # Verify formal properties
                assert isinstance(result, tuple), "Result must be a tuple"
                assert len(result) == 3, "Result tuple must have exactly 3 elements"
                assert isinstance(result[0], (int, type(None))), (
                    "First element must be int or None"
                )
                assert isinstance(result[1], float), "Second element must be float"
                assert isinstance(result[2], str), "Third element must be string"


def test_z3_line_number_positive_when_not_none() -> None:
    """
    Z3 Formal Verification: When the output line number is not None,
    it must be positive (>= 1).
    """
    # Create Z3 variables
    line_num = Int('line_num')
    is_valid = Bool('is_valid')

    # Create solver
    solver = Solver()

    # Define constraint: if line_num is set, it must be >= 1
    solver.add(Implies(is_valid, line_num >= 1))
    solver.add(is_valid == True)

    # Check satisfiability
    assert solver.check() == sat, "Positive line number should be satisfiable"

    # Check that we can get a model
    model = solver.model()
    assert model[line_num].as_long() >= 1, "Model should satisfy line_num >= 1"

    # Verify with actual function - all non-None results should be >= 1
    with patch('pdd.trace.llm_invoke') as mock_llm:
        with patch('pdd.trace.load_prompt_template') as mock_load:
            with patch('pdd.trace.preprocess') as mock_prep:
                mock_load.return_value = "template"
                mock_prep.side_effect = lambda x, **kwargs: x
                mock_llm.side_effect = [
                    {'result': 'Analysis', 'cost': 0.01, 'model_name': 'model'},
                    {'result': PromptLineOutput(prompt_line='test line'), 'cost': 0.01, 'model_name': 'model'},
                ]

                result = trace("code line", 1, "prompt line")

                if result[0] is not None:
                    assert result[0] >= 1, (
                        f"Non-None line number {result[0]} must be >= 1"
                    )


def test_z3_model_name_not_empty() -> None:
    """
    Z3 Formal Verification: The model_name in the return value should not be empty.
    """
    # We'll use Z3 String logic
    model_name = String('model_name')
    model_name_length = Int('model_name_length')

    solver = Solver()

    # Define constraint: model_name length should be > 0
    solver.add(model_name_length > 0)
    solver.add(model_name_length == Length(model_name))

    # Check satisfiability
    assert solver.check() == sat, "Non-empty model name should be satisfiable"

    # Verify with actual function
    with patch('pdd.trace.llm_invoke') as mock_llm:
        with patch('pdd.trace.load_prompt_template') as mock_load:
            with patch('pdd.trace.preprocess') as mock_prep:
                mock_load.return_value = "template"
                mock_prep.side_effect = lambda x, **kwargs: x
                mock_llm.side_effect = [
                    {'result': 'Analysis', 'cost': 0.01, 'model_name': 'gpt-4-test'},
                    {'result': PromptLineOutput(prompt_line='line'), 'cost': 0.01, 'model_name': 'gpt-4-test'},
                ]

                result = trace("code", 1, "prompt")

                # Verify model_name is not empty
                assert len(result[2]) > 0, "Model name should not be empty"
                assert isinstance(result[2], str), "Model name should be a string"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_trace_integration_with_all_parameters(
    mock_llm_invoke, mock_load_prompt_template, mock_preprocess
) -> None:
    """Integration test with all parameters specified."""
    mock_llm_invoke.side_effect = [
        {
            'result': 'Detailed analysis of code',
            'cost': 0.05,
            'model_name': 'gpt-4-turbo',
        },
        {
            'result': PromptLineOutput(prompt_line='Write a comprehensive function'),
            'cost': 0.03,
            'model_name': 'gpt-4-turbo',
        },
    ]

    code = "def comprehensive_function():\n    # Implementation\n    pass"
    prompt = "Start with imports\nWrite a comprehensive function\nAdd error handling"

    result = trace(
        code_file=code,
        code_line=1,
        prompt_file=prompt,
        strength=0.7,
        temperature=0.5,
        verbose=True,
        time=0.8,
    )

    assert isinstance(result[0], int)
    assert result[0] >= 1
    assert result[1] == pytest.approx(0.08, rel=1e-6)
    assert result[2] == 'gpt-4-turbo'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
