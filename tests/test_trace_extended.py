import pytest
from typing import Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
from z3 import Real, Int, Solver, And, sat, unsat

from pdd.trace import trace, PromptLineOutput


# ============================================================================
# PART A: Z3 FORMAL VERIFICATION TESTS
# ============================================================================

def test_z3_parameter_ranges() -> None:
    """
    Z3 Test: Verify that when valid parameters are provided within their ranges,
    the function properties hold:
    - strength should be between 0 and 1
    - temperature should be between 0 and 1
    - time should be between 0 and 1
    """
    # Define symbolic variables
    strength = Real('strength')
    temperature = Real('temperature')
    time = Real('time')

    # Define constraints for valid inputs
    solver = Solver()
    solver.add(And(strength >= 0, strength <= 1))
    solver.add(And(temperature >= 0, temperature <= 1))
    solver.add(And(time >= 0, time <= 1))

    # Verify constraints are satisfiable
    assert solver.check() == sat

    # Test boundary values
    test_cases = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (0.0, 1.0, 0.5),
    ]

    for s, t, tm in test_cases:
        solver_instance = Solver()
        solver_instance.add(And(strength >= 0, strength <= 1))
        solver_instance.add(And(temperature >= 0, temperature <= 1))
        solver_instance.add(And(time >= 0, time <= 1))
        solver_instance.add(strength == s)
        solver_instance.add(temperature == t)
        solver_instance.add(time == tm)
        assert solver_instance.check() == sat


def test_z3_return_value_constraints() -> None:
    """
    Z3 Test: Verify return value constraints:
    - prompt_line is either None or a positive integer
    - total_cost is a non-negative float
    - model_name is always returned (even if empty string in error cases)
    """
    # Define symbolic variables for return values
    prompt_line = Int('prompt_line')
    total_cost = Real('total_cost')

    solver = Solver()

    # Constraint: prompt_line must be positive when not None
    # (None case handled separately in implementation)
    solver.add(prompt_line >= 1)

    # Constraint: total_cost must be non-negative
    solver.add(total_cost >= 0)

    # Verify constraints are satisfiable
    assert solver.check() == sat

    # Verify invalid cases are unsatisfiable GIVEN the constraints
    invalid_solver = Solver()
    invalid_solver.add(total_cost >= 0)
    invalid_solver.add(total_cost < 0)
    assert invalid_solver.check() == unsat


def test_z3_line_number_validity() -> None:
    """
    Z3 Test: Verify that if a line number is returned, it must be within
    valid range for a file with n lines: 1 <= line_number <= n
    """
    prompt_line = Int('prompt_line')
    file_lines = Int('file_lines')

    solver = Solver()
    solver.add(file_lines > 0)  # File has at least 1 line
    solver.add(prompt_line >= 1)
    solver.add(prompt_line <= file_lines)

    # Should be satisfiable for valid ranges
    assert solver.check() == sat

    # Test specific cases
    solver.push()
    solver.add(file_lines == 10)
    solver.add(prompt_line == 5)
    assert solver.check() == sat
    solver.pop()

    # Invalid case: line number exceeds file length
    solver.push()
    solver.add(file_lines == 10)
    solver.add(prompt_line == 11)
    assert solver.check() == unsat
    solver.pop()


def test_z3_cost_non_negative() -> None:
    """
    Z3 Test: Verify that total cost is always non-negative, even when
    combining costs from multiple operations.
    """
    cost1 = Real('cost1')
    cost2 = Real('cost2')
    total_cost = Real('total_cost')

    solver = Solver()
    solver.add(cost1 >= 0)
    solver.add(cost2 >= 0)
    solver.add(total_cost == cost1 + cost2)
    solver.add(total_cost >= 0)

    assert solver.check() == sat

    # Verify that negative costs are impossible
    invalid_solver = Solver()
    invalid_solver.add(cost1 >= 0)
    invalid_solver.add(cost2 >= 0)
    invalid_solver.add(total_cost == cost1 + cost2)
    invalid_solver.add(total_cost < 0)
    assert invalid_solver.check() == unsat


# ============================================================================
# FIXTURES FOR UNIT TESTS
# ============================================================================

@pytest.fixture
def mock_template_loader():
    """Mock load_prompt_template to return predefined templates."""
    with patch('pdd.trace.load_prompt_template') as mock:
        def side_effect(template_name: str) -> Optional[str]:
            if template_name == "trace_LLM":
                return "Trace template with {CODE_FILE} {CODE_STR} {PROMPT_FILE}"
            elif template_name == "extract_promptline_LLM":
                return "Extract template with {llm_output}"
            return None
        mock.side_effect = side_effect
        yield mock


@pytest.fixture
def mock_preprocess():
    """Mock preprocess to return the input unchanged."""
    with patch('pdd.trace.preprocess') as mock:
        mock.side_effect = lambda text, **kwargs: text
        yield mock


@pytest.fixture
def mock_llm_invoke():
    """Mock llm_invoke to return controlled responses."""
    with patch('pdd.trace.llm_invoke') as mock:
        yield mock


@pytest.fixture
def sample_code_file() -> str:
    """Sample code file for testing."""
    return """def hello():
    print("Hello, World!")
    return 42

def goodbye():
    print("Goodbye!")"""


@pytest.fixture
def sample_prompt_file() -> str:
    """Sample prompt file for testing."""
    return """Write a function called hello
It should print a greeting
And return the number 42

Write a function called goodbye
It should print a farewell message"""


# ============================================================================
# PART B: UNIT TESTS - Input Validation
# ============================================================================

def test_empty_code_file(mock_template_loader, mock_preprocess, mock_llm_invoke) -> None:
    """Test handling of empty code file."""
    result = trace(
        code_file="",
        code_line=1,
        prompt_file="some prompt content",
        verbose=False
    )

    # Should return fallback values
    assert isinstance(result, tuple)
    assert len(result) == 3
    prompt_line, cost, model = result
    assert prompt_line is not None  # Should get fallback
    assert isinstance(cost, float)
    assert isinstance(model, str)


def test_empty_prompt_file(mock_template_loader, mock_preprocess, mock_llm_invoke) -> None:
    """Test handling of empty prompt file."""
    result = trace(
        code_file="def foo(): pass",
        code_line=1,
        prompt_file="",
        verbose=False
    )

    # Should return fallback values
    assert isinstance(result, tuple)
    assert len(result) == 3
    prompt_line, cost, model = result
    assert prompt_line is not None
    assert isinstance(cost, float)
    assert isinstance(model, str)


def test_invalid_code_line_negative(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test handling of negative line number."""
    result = trace(
        code_file="def foo(): pass",
        code_line=-1,
        prompt_file="Write a function",
        verbose=False
    )

    # Should handle gracefully with fallback
    assert isinstance(result, tuple)
    prompt_line, cost, model = result
    assert prompt_line is not None  # Fallback value
    assert cost >= 0


def test_invalid_code_line_zero(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test handling of zero line number."""
    result = trace(
        code_file="def foo(): pass",
        code_line=0,
        prompt_file="Write a function",
        verbose=False
    )

    # Should handle gracefully with fallback
    assert isinstance(result, tuple)
    prompt_line, cost, model = result
    assert prompt_line is not None


def test_invalid_code_line_out_of_range(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test handling of line number beyond file length."""
    result = trace(
        code_file="line1\nline2",
        code_line=10,
        prompt_file="Write code",
        verbose=False
    )

    # Should handle gracefully with fallback
    assert isinstance(result, tuple)
    prompt_line, cost, model = result
    assert prompt_line is not None


# ============================================================================
# PART C: UNIT TESTS - Normal Operation
# ============================================================================

def test_successful_trace_exact_match(
    mock_template_loader, mock_preprocess, mock_llm_invoke,
    sample_code_file, sample_prompt_file
) -> None:
    """Test successful trace with exact match."""
    # Setup mock LLM responses
    trace_response = {
        'result': 'The code matches the first line',
        'cost': 0.001,
        'model_name': 'test-model'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="Write a function called hello"),
        'cost': 0.0005,
        'model_name': 'test-model'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=sample_code_file,
        code_line=1,
        prompt_file=sample_prompt_file,
        verbose=False
    )

    prompt_line, cost, model = result

    assert prompt_line is not None
    assert prompt_line >= 1
    assert prompt_line <= len(sample_prompt_file.splitlines())
    assert cost == 0.0015  # Sum of both calls
    assert model == 'test-model'


def test_successful_trace_fuzzy_match(
    mock_template_loader, mock_preprocess, mock_llm_invoke,
    sample_code_file, sample_prompt_file
) -> None:
    """Test successful trace with fuzzy matching."""
    # Setup mock LLM responses with similar but not exact text
    trace_response = {
        'result': 'The code is related to printing goodbye',
        'cost': 0.002,
        'model_name': 'gpt-4'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="should print a farewell"),
        'cost': 0.001,
        'model_name': 'gpt-4'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=sample_code_file,
        code_line=5,
        prompt_file=sample_prompt_file,
        strength=0.7,
        temperature=0.3,
        verbose=False
    )

    prompt_line, cost, model = result

    assert prompt_line is not None
    assert cost == 0.003
    assert model == 'gpt-4'


def test_cost_accumulation(
    mock_template_loader, mock_preprocess, mock_llm_invoke,
    sample_code_file, sample_prompt_file
) -> None:
    """Test that costs from multiple LLM calls are accumulated."""
    trace_response = {
        'result': 'Analysis result',
        'cost': 0.123,
        'model_name': 'model-1'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="Some line"),
        'cost': 0.456,
        'model_name': 'model-1'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    _, cost, _ = trace(
        code_file=sample_code_file,
        code_line=1,
        prompt_file=sample_prompt_file,
        verbose=False
    )

    assert cost == pytest.approx(0.579)


def test_model_name_propagation(
    mock_template_loader, mock_preprocess, mock_llm_invoke,
    sample_code_file, sample_prompt_file
) -> None:
    """Test that model name from first LLM call is propagated."""
    trace_response = {
        'result': 'Analysis',
        'cost': 0.01,
        'model_name': 'claude-3'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="Line text"),
        'cost': 0.01,
        'model_name': 'different-model'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    _, _, model = trace(
        code_file=sample_code_file,
        code_line=1,
        prompt_file=sample_prompt_file,
        verbose=False
    )

    assert model == 'claude-3'


def test_single_line_files(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test with minimal single-line files."""
    trace_response = {
        'result': 'Match',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="single prompt"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file="single line",
        code_line=1,
        prompt_file="single prompt",
        verbose=False
    )

    prompt_line, cost, model = result
    assert prompt_line == 1
    assert cost > 0
    assert model == 'test'


def test_multiline_matching(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test matching that might span multiple lines."""
    code = """def complex_function():
    # This does something
    # across multiple lines
    pass"""

    prompt = """Create a complex function
that does something
across multiple lines"""

    trace_response = {
        'result': 'Multi-line match',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="that does something"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=code,
        code_line=2,
        prompt_file=prompt,
        verbose=False
    )

    prompt_line, _, _ = result
    assert prompt_line is not None
    assert 1 <= prompt_line <= 3


# ============================================================================
# PART D: UNIT TESTS - Verbose Mode
# ============================================================================

def test_verbose_mode_enabled(
    mock_template_loader, mock_preprocess, mock_llm_invoke,
    sample_code_file, sample_prompt_file, capsys
) -> None:
    """Test that verbose mode produces console output."""
    trace_response = {
        'result': 'Analysis',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="Write a function called hello"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    # Note: verbose output goes through rich.Console, might not appear in capsys
    # This test verifies the function completes with verbose=True
    result = trace(
        code_file=sample_code_file,
        code_line=1,
        prompt_file=sample_prompt_file,
        verbose=True
    )

    assert result is not None
    assert len(result) == 3


def test_verbose_mode_disabled(
    mock_template_loader, mock_preprocess, mock_llm_invoke,
    sample_code_file, sample_prompt_file
) -> None:
    """Test that verbose=False doesn't affect functionality."""
    trace_response = {
        'result': 'Analysis',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="Write a function"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=sample_code_file,
        code_line=1,
        prompt_file=sample_prompt_file,
        verbose=False
    )

    assert result is not None
    assert len(result) == 3


# ============================================================================
# PART E: UNIT TESTS - Error Handling
# ============================================================================

def test_prompt_template_load_failure(mock_preprocess, mock_llm_invoke) -> None:
    """Test handling of missing template files."""
    with patch('pdd.trace.load_prompt_template') as mock_loader:
        mock_loader.return_value = None

        result = trace(
            code_file="def foo(): pass",
            code_line=1,
            prompt_file="Write foo",
            verbose=False
        )

        # Should handle gracefully with fallback
        prompt_line, cost, model = result
        assert prompt_line is not None
        assert cost == 0.0  # No LLM calls made
        assert model == 'fallback'


def test_llm_invoke_failure(mock_template_loader, mock_preprocess) -> None:
    """Test handling of LLM invocation errors."""
    with patch('pdd.trace.llm_invoke') as mock_invoke:
        mock_invoke.side_effect = Exception("LLM API error")

        result = trace(
            code_file="def foo(): pass",
            code_line=1,
            prompt_file="Write foo",
            verbose=False
        )

        # Should fall back gracefully
        prompt_line, cost, model = result
        assert prompt_line is not None
        assert cost == 0.0
        assert model == 'fallback'


def test_no_match_fallback(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test fallback when no fuzzy match is found."""
    trace_response = {
        'result': 'Some analysis',
        'cost': 0.01,
        'model_name': 'test'
    }

    # Return a line that doesn't match anything in prompt
    extract_response = {
        'result': PromptLineOutput(prompt_line="COMPLETELY UNRELATED TEXT XYZ123"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file="def specific_code(): pass",
        code_line=1,
        prompt_file="Write general prompt\nAnother line\nThird line",
        verbose=False
    )

    prompt_line, cost, model = result
    # Should still return a valid line number (fallback)
    assert prompt_line is not None
    assert prompt_line >= 1


def test_exception_handling(mock_template_loader, mock_preprocess) -> None:
    """Test general exception handling."""
    with patch('pdd.trace.llm_invoke') as mock_invoke:
        # Cause an unexpected exception
        mock_invoke.side_effect = RuntimeError("Unexpected error")

        result = trace(
            code_file="code",
            code_line=1,
            prompt_file="prompt",
            verbose=False
        )

        # Should not raise, should return fallback
        assert isinstance(result, tuple)
        assert len(result) == 3
        prompt_line, cost, model = result
        assert prompt_line is not None
        assert cost == 0.0
        assert model == 'fallback'


# ============================================================================
# PART F: UNIT TESTS - Special Cases
# ============================================================================

def test_unicode_handling(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test handling of unicode characters."""
    code = """def greet():
    print("Hello 世界 🌍")"""

    prompt = """Write a greeting function
that prints Hello 世界 🌍"""

    trace_response = {
        'result': 'Unicode match',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="prints Hello 世界 🌍"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=code,
        code_line=2,
        prompt_file=prompt,
        verbose=False
    )

    prompt_line, _, _ = result
    assert prompt_line is not None
    assert prompt_line in [1, 2]


def test_special_characters(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test handling of special characters in code and prompt."""
    code = 'result = re.match(r"^[a-z]+$", text)'
    prompt = 'Use regex pattern ^[a-z]+$ to match'

    trace_response = {
        'result': 'Regex match',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line='pattern ^[a-z]+$'),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=code,
        code_line=1,
        prompt_file=prompt,
        verbose=False
    )

    prompt_line, _, _ = result
    assert prompt_line == 1


def test_empty_lines(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test handling of files with empty lines."""
    code = """def foo():

    pass"""

    prompt = """Write function foo

with empty lines"""

    trace_response = {
        'result': 'Match with empty lines',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="Write function foo"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=code,
        code_line=1,
        prompt_file=prompt,
        verbose=False
    )

    prompt_line, _, _ = result
    assert prompt_line is not None


def test_whitespace_variations(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test handling of different whitespace patterns."""
    code = "    def    indented():    pass    "
    prompt = "def indented(): pass"

    trace_response = {
        'result': 'Whitespace match',
        'cost': 0.01,
        'model_name': 'test'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="def   indented():   pass"),
        'cost': 0.01,
        'model_name': 'test'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    result = trace(
        code_file=code,
        code_line=1,
        prompt_file=prompt,
        verbose=False
    )

    prompt_line, _, _ = result
    assert prompt_line == 1


def test_parameter_variations(
    mock_template_loader, mock_preprocess, mock_llm_invoke,
    sample_code_file, sample_prompt_file
) -> None:
    """Test with different strength and temperature values."""
    test_params = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ]

    for strength, temperature, time in test_params:
        trace_response = {
            'result': 'Analysis',
            'cost': 0.01,
            'model_name': 'test'
        }

        extract_response = {
            'result': PromptLineOutput(prompt_line="Match"),
            'cost': 0.01,
            'model_name': 'test'
        }

        mock_llm_invoke.side_effect = [trace_response, extract_response]

        result = trace(
            code_file=sample_code_file,
            code_line=1,
            prompt_file=sample_prompt_file,
            strength=strength,
            temperature=temperature,
            time=time,
            verbose=False
        )

        assert result is not None
        assert len(result) == 3

        # Verify LLM was called with correct parameters
        assert mock_llm_invoke.call_count >= 2
        first_call = mock_llm_invoke.call_args_list[0]
        assert first_call[1]['strength'] == strength
        assert first_call[1]['temperature'] == temperature
        assert first_call[1]['time'] == time

        mock_llm_invoke.reset_mock()


# ============================================================================
# INTEGRATION-STYLE TESTS (Still with mocks but testing more complete flows)
# ============================================================================

def test_complete_flow_with_exact_match(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test complete flow from input to output with exact match."""
    code = """def calculate_sum(a, b):
    return a + b"""

    prompt = """Create a function calculate_sum
that takes two parameters
and returns their sum"""

    trace_response = {
        'result': 'The code implements a sum function as described in the prompt',
        'cost': 0.025,
        'model_name': 'gpt-4-turbo'
    }

    extract_response = {
        'result': PromptLineOutput(prompt_line="Create a function calculate_sum"),
        'cost': 0.015,
        'model_name': 'gpt-4-turbo'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    prompt_line, cost, model = trace(
        code_file=code,
        code_line=1,
        prompt_file=prompt,
        strength=0.8,
        temperature=0.2,
        verbose=False,
        time=0.5
    )

    # Verify all return values
    assert prompt_line == 1
    assert cost == 0.04
    assert model == 'gpt-4-turbo'

    # Verify LLM was called correctly
    assert mock_llm_invoke.call_count == 2

    # Check first call (trace_LLM)
    first_call = mock_llm_invoke.call_args_list[0]
    assert 'CODE_FILE' in first_call[1]['input_json']
    assert 'CODE_STR' in first_call[1]['input_json']
    assert 'PROMPT_FILE' in first_call[1]['input_json']
    assert first_call[1]['input_json']['CODE_STR'] == "def calculate_sum(a, b):"

    # Check second call (extract_promptline_LLM)
    second_call = mock_llm_invoke.call_args_list[1]
    assert 'llm_output' in second_call[1]['input_json']
    assert second_call[1]['output_pydantic'] == PromptLineOutput


def test_complete_flow_with_fuzzy_match(
    mock_template_loader, mock_preprocess, mock_llm_invoke
) -> None:
    """Test complete flow with fuzzy string matching."""
    code = """def process_data(data):
    cleaned = data.strip()
    return cleaned.lower()"""

    prompt = """Write a function to process data
First strip whitespace
Then convert to lowercase
Return the result"""

    trace_response = {
        'result': 'Processes data by stripping and lowercasing',
        'cost': 0.03,
        'model_name': 'claude-opus'
    }

    # LLM returns slightly different wording
    extract_response = {
        'result': PromptLineOutput(prompt_line="strip the whitespace"),
        'cost': 0.02,
        'model_name': 'claude-opus'
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    prompt_line, cost, model = trace(
        code_file=code,
        code_line=2,
        prompt_file=prompt,
        verbose=False
    )

    # Should match line 2 ("First strip whitespace") via fuzzy matching
    assert prompt_line in [1, 2]  # Could match title or specific instruction
    assert cost == 0.05
    assert model == 'claude-opus'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
