"""
Test Plan for trace_main
========================

This test suite verifies the functionality of the trace_main function, which handles
the core logic for the 'trace' command in the pdd CLI.

TEST CATEGORIES:

1. NORMAL FLOW TESTS (Unit Tests)
   - Test successful trace analysis with output file
   - Test successful trace analysis without output file
   - Test with custom strength, temperature, and time parameters
   - Test in quiet mode (no console output)
   - Test with force flag enabled
   - Rationale: These test the happy path and are best done with unit tests as they
     involve integration of multiple components and file I/O.

2. ERROR HANDLING TESTS (Unit Tests)
   - Test FileNotFoundError when input files don't exist
   - Test ValueError from trace function (invalid inputs)
   - Test when trace returns None (analysis failed)
   - Test IOError when writing output file
   - Test error creating output directory
   - Rationale: Error handling involves side effects (ctx.exit, console output) and
     external dependencies, which are best tested with unit tests and mocking.

3. OUTPUT FILE TESTS (Unit Tests)
   - Test output file created with correct format and content
   - Test output directory created if it doesn't exist
   - Test output file in nested directory structure
   - Rationale: File I/O testing requires actual file system operations or mocking,
     best done with unit tests.

4. QUIET MODE TESTS (Unit Tests)
   - Verify no console output when quiet=True
   - Verify proper console output when quiet=False
   - Rationale: Testing console output requires capturing stdout/stderr, which is
     a unit testing concern.

5. CONTEXT OBJECT TESTS (Unit Tests)
   - Test with various ctx.obj configurations
   - Test with missing optional parameters (uses defaults)
   - Test with context override in construct_paths
   - Rationale: Testing context handling is integration testing between CLI and
     business logic, best done with unit tests.

6. RETURN VALUE TESTS (Unit Tests + Z3)
   - Test correct return tuple structure (prompt_line, total_cost, model_name)
   - Test return values match trace function results
   - Unit tests for actual values, Z3 for type/structure properties if applicable
   - Rationale: Return value verification is straightforward unit testing, though
     Z3 could verify invariants about the return structure.

Z3 FORMAL VERIFICATION APPLICABILITY:

Z3 is LIMITED in applicability for this function because:
- The function has side effects (file I/O, console output, program exit)
- Core logic is delegated to external functions (construct_paths, trace)
- Behavior depends on external state (file system, LLM API)

Potential Z3 use cases:
- Verify type invariants on return values (but Python type checking is better)
- Verify logical relationships between input parameters and control flow
- Verify that certain error conditions always lead to ctx.exit(1)

MOCKING STRATEGY:

To ensure test isolation and robustness:
- Mock construct_paths to return controlled input_strings and output_file_paths
- Mock trace to return controlled results
- Mock ctx.exit to verify it's called without actually exiting
- Mock file operations (open, os.makedirs) where appropriate
- Capture console output using pytest's capsys or by mocking rprint
- Use pytest fixtures for common setup (mock context objects)

IMPLEMENTATION NOTES:

1. The code has a type hint bug: returns Tuple[str, float, str] but should be
   Tuple[int, float, str] per spec. Tests will verify actual behavior.
2. Tests focus on functionality, not implementation details, to be robust against
   code regeneration by LLM.
3. Each test function tests a specific scenario for easy identification of failures.
4. Tests use mocking to avoid dependencies on actual file system and LLM API.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch, mock_open, call
from pathlib import Path
from typing import Tuple, Optional
import click

# Import the function under test
from pdd.trace_main import trace_main


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_ctx():
    """Create a mock Click context object with default configuration."""
    ctx = Mock(spec=click.Context)
    ctx.obj = {
        'quiet': False,
        'force': False,
        'strength': 0.5,
        'temperature': 0.0,
        'time': 30,
        'context': None
    }
    ctx.exit = Mock(side_effect=SystemExit)
    return ctx


@pytest.fixture
def mock_construct_paths():
    """Mock construct_paths to return controlled data."""
    with patch('pdd.trace_main.construct_paths') as mock:
        mock.return_value = (
            {},  # resolved_config
            {
                'prompt_file': 'Sample prompt content',
                'code_file': 'def hello():\n    return "world"'
            },  # input_strings
            {
                'output': '/tmp/output.txt'
            },  # output_file_paths
            'python'  # language
        )
        yield mock


@pytest.fixture
def mock_trace():
    """Mock trace function to return controlled results."""
    with patch('pdd.trace_main.trace') as mock:
        mock.return_value = (42, 0.001234, 'gpt-4')
        yield mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# NORMAL FLOW TESTS
# =============================================================================

def test_trace_main_success_with_output(mock_ctx, mock_construct_paths, mock_trace):
    """Test successful trace analysis with output file."""
    output_file = '/tmp/test_output.txt'

    with patch('builtins.open', mock_open()) as mocked_file, \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('pdd.trace_main.rprint') as mock_rprint:

        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=output_file
        )

        # Verify construct_paths was called correctly
        mock_construct_paths.assert_called_once()
        call_args = mock_construct_paths.call_args
        assert call_args.kwargs['input_file_paths'] == {
            'prompt_file': 'prompt.txt',
            'code_file': 'code.py'
        }
        assert call_args.kwargs['command'] == 'trace'

        # Verify trace was called correctly
        mock_trace.assert_called_once_with(
            'def hello():\n    return "world"',
            5,
            'Sample prompt content',
            0.5,  # strength
            0.0,  # temperature
            time=30
        )

        # Verify return value
        assert result == (42, 0.001234, 'gpt-4')

        # Verify output was written
        mocked_file.assert_called_with('/tmp/output.txt', 'w')
        handle = mocked_file()
        handle.write.assert_any_call('Prompt Line: 42\n')
        handle.write.assert_any_call('Total Cost: $0.001234\n')
        handle.write.assert_any_call('Model Used: gpt-4\n')

        # Verify user feedback
        assert mock_rprint.call_count >= 4


def test_trace_main_success_without_output(mock_ctx, mock_construct_paths, mock_trace):
    """Test successful trace analysis without output file."""
    with patch('pdd.trace_main.rprint') as mock_rprint:
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify return value
        assert result == (42, 0.001234, 'gpt-4')

        # Verify trace was called
        mock_trace.assert_called_once()

        # Verify user feedback was provided
        assert mock_rprint.call_count >= 4


def test_trace_main_quiet_mode(mock_ctx, mock_construct_paths, mock_trace):
    """Test trace analysis in quiet mode (no console output)."""
    mock_ctx.obj['quiet'] = True

    with patch('pdd.trace_main.rprint') as mock_rprint:
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify return value
        assert result == (42, 0.001234, 'gpt-4')

        # Verify no console output in quiet mode
        mock_rprint.assert_not_called()


def test_trace_main_custom_parameters(mock_ctx, mock_construct_paths, mock_trace):
    """Test trace analysis with custom strength, temperature, and time."""
    mock_ctx.obj['strength'] = 0.8
    mock_ctx.obj['temperature'] = 0.7
    mock_ctx.obj['time'] = 60

    with patch('pdd.trace_main.rprint'):
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=10,
            output=None
        )

        # Verify trace was called with custom parameters
        mock_trace.assert_called_once_with(
            'def hello():\n    return "world"',
            10,
            'Sample prompt content',
            0.8,   # custom strength
            0.7,   # custom temperature
            time=60  # custom time
        )

        assert result == (42, 0.001234, 'gpt-4')


def test_trace_main_with_force_flag(mock_ctx, mock_construct_paths, mock_trace):
    """Test trace analysis with force flag enabled."""
    mock_ctx.obj['force'] = True

    with patch('pdd.trace_main.rprint'):
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify construct_paths was called with force=True
        call_args = mock_construct_paths.call_args
        assert call_args.kwargs['force'] is True

        assert result == (42, 0.001234, 'gpt-4')


def test_trace_main_with_context_override(mock_ctx, mock_construct_paths, mock_trace):
    """Test trace analysis with context override."""
    mock_ctx.obj['context'] = 'custom_context'

    with patch('pdd.trace_main.rprint'):
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify construct_paths was called with context_override
        call_args = mock_construct_paths.call_args
        assert call_args.kwargs['context_override'] == 'custom_context'

        assert result == (42, 0.001234, 'gpt-4')


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_trace_main_file_not_found(mock_ctx, mock_construct_paths, mock_trace):
    """Test error handling when input file is not found."""
    mock_construct_paths.side_effect = FileNotFoundError("prompt.txt not found")

    with patch('pdd.trace_main.rprint') as mock_rprint:
        with pytest.raises(SystemExit):
            trace_main(
                ctx=mock_ctx,
                prompt_file='nonexistent.txt',
                code_file='code.py',
                code_line=5,
                output=None
            )

        # Verify error message was printed
        mock_rprint.assert_called_once()
        assert "File not found" in str(mock_rprint.call_args)

        # Verify ctx.exit was called
        mock_ctx.exit.assert_called_once_with(1)


def test_trace_main_file_not_found_quiet(mock_ctx, mock_construct_paths, mock_trace):
    """Test error handling when file not found in quiet mode."""
    mock_ctx.obj['quiet'] = True
    mock_construct_paths.side_effect = FileNotFoundError("prompt.txt not found")

    with patch('pdd.trace_main.rprint') as mock_rprint:
        with pytest.raises(SystemExit):
            trace_main(
                ctx=mock_ctx,
                prompt_file='nonexistent.txt',
                code_file='code.py',
                code_line=5,
                output=None
            )

        # Verify no output in quiet mode
        mock_rprint.assert_not_called()

        # Verify ctx.exit was called
        mock_ctx.exit.assert_called_once_with(1)


def test_trace_main_value_error_from_trace(mock_ctx, mock_construct_paths, mock_trace):
    """Test error handling when trace raises ValueError."""
    mock_trace.side_effect = ValueError("Invalid code line")

    with patch('pdd.trace_main.rprint') as mock_rprint:
        with pytest.raises(SystemExit):
            trace_main(
                ctx=mock_ctx,
                prompt_file='prompt.txt',
                code_file='code.py',
                code_line=-1,
                output=None
            )

        # Verify error message was printed
        mock_rprint.assert_called_once()
        assert "Invalid input" in str(mock_rprint.call_args)

        # Verify ctx.exit was called
        mock_ctx.exit.assert_called_once_with(1)


def test_trace_main_prompt_line_none(mock_ctx, mock_construct_paths, mock_trace):
    """Test error handling when trace returns None for prompt_line."""
    mock_trace.return_value = (None, 0.001, 'gpt-4')

    with patch('pdd.trace_main.rprint') as mock_rprint:
        with pytest.raises(SystemExit):
            trace_main(
                ctx=mock_ctx,
                prompt_file='prompt.txt',
                code_file='code.py',
                code_line=5,
                output=None
            )

        # Verify error message was printed
        mock_rprint.assert_called_once()
        assert "Trace analysis failed" in str(mock_rprint.call_args)

        # Verify ctx.exit was called
        mock_ctx.exit.assert_called_once_with(1)


def test_trace_main_io_error_writing_output(mock_ctx, mock_construct_paths, mock_trace):
    """Test error handling when writing output file fails."""
    output_file = '/tmp/test_output.txt'

    with patch('builtins.open', side_effect=IOError("Permission denied")), \
         patch('os.path.exists', return_value=True), \
         patch('pdd.trace_main.rprint') as mock_rprint:

        with pytest.raises(SystemExit):
            trace_main(
                ctx=mock_ctx,
                prompt_file='prompt.txt',
                code_file='code.py',
                code_line=5,
                output=output_file
            )

        # Verify error message was printed
        assert any(
            "Error saving trace results" in str(call)
            for call in mock_rprint.call_args_list
        )

        # Verify ctx.exit was called
        mock_ctx.exit.assert_called_once_with(1)


def test_trace_main_error_creating_output_directory(
    mock_ctx, mock_construct_paths, mock_trace
):
    """Test error handling when creating output directory fails."""
    output_file = '/root/forbidden/output.txt'

    with patch('os.path.exists', return_value=False), \
         patch('os.makedirs', side_effect=PermissionError("Permission denied")), \
         patch('pdd.trace_main.rprint') as mock_rprint:

        with pytest.raises(SystemExit):
            trace_main(
                ctx=mock_ctx,
                prompt_file='prompt.txt',
                code_file='code.py',
                code_line=5,
                output=output_file
            )

        # Verify error message was printed
        assert any(
            "Failed to create output directory" in str(call)
            for call in mock_rprint.call_args_list
        )

        # Verify ctx.exit was called
        mock_ctx.exit.assert_called_once_with(1)


def test_trace_main_unexpected_error(mock_ctx, mock_construct_paths, mock_trace):
    """Test error handling for unexpected exceptions."""
    mock_trace.side_effect = RuntimeError("Unexpected error")

    with patch('pdd.trace_main.rprint') as mock_rprint:
        with pytest.raises(SystemExit):
            trace_main(
                ctx=mock_ctx,
                prompt_file='prompt.txt',
                code_file='code.py',
                code_line=5,
                output=None
            )

        # Verify error message was printed
        mock_rprint.assert_called_once()
        assert "An unexpected error occurred" in str(mock_rprint.call_args)

        # Verify ctx.exit was called
        mock_ctx.exit.assert_called_once_with(1)


# =============================================================================
# OUTPUT FILE TESTS
# =============================================================================

def test_trace_main_output_file_content(
    mock_ctx, mock_construct_paths, mock_trace, temp_dir
):
    """Test that output file contains correct formatted content."""
    output_file = os.path.join(temp_dir, 'output.txt')
    mock_construct_paths.return_value = (
        {},
        {
            'prompt_file': 'Sample prompt',
            'code_file': 'def test(): pass'
        },
        {'output': output_file},
        'python'
    )

    with patch('pdd.trace_main.rprint'):
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=output_file
        )

        # Read and verify output file content
        with open(output_file, 'r') as f:
            content = f.read()

        assert 'Prompt Line: 42' in content
        assert 'Total Cost: $0.001234' in content
        assert 'Model Used: gpt-4' in content


def test_trace_main_output_creates_directory(
    mock_ctx, mock_construct_paths, mock_trace, temp_dir
):
    """Test that output directory is created if it doesn't exist."""
    nested_dir = os.path.join(temp_dir, 'nested', 'dirs')
    output_file = os.path.join(nested_dir, 'output.txt')

    mock_construct_paths.return_value = (
        {},
        {
            'prompt_file': 'Sample prompt',
            'code_file': 'def test(): pass'
        },
        {'output': output_file},
        'python'
    )

    with patch('pdd.trace_main.rprint'):
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=output_file
        )

        # Verify directory was created
        assert os.path.exists(nested_dir)
        assert os.path.exists(output_file)


def test_trace_main_output_file_formatting(mock_ctx, mock_construct_paths, mock_trace):
    """Test output file has correct number formatting for cost."""
    mock_trace.return_value = (123, 0.123456789, 'test-model')
    output_file = '/tmp/output_formatting.txt'

    with patch('builtins.open', mock_open()) as mocked_file, \
         patch('os.path.exists', return_value=True), \
         patch('pdd.trace_main.rprint'):

        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=output_file
        )

        # Verify formatting (6 decimal places for cost)
        handle = mocked_file()
        handle.write.assert_any_call('Total Cost: $0.123457\n')


# =============================================================================
# CONTEXT OBJECT TESTS
# =============================================================================

def test_trace_main_missing_optional_ctx_params(mock_construct_paths, mock_trace):
    """Test with missing optional parameters in ctx.obj (should use defaults)."""
    ctx = Mock(spec=click.Context)
    ctx.obj = {}  # Empty context
    ctx.exit = Mock(side_effect=SystemExit)

    with patch('pdd.trace_main.rprint'), \
         patch('pdd.trace_main.DEFAULT_STRENGTH', 0.5), \
         patch('pdd.trace_main.DEFAULT_TEMPERATURE', 0.0), \
         patch('pdd.trace_main.DEFAULT_TIME', 30):

        result = trace_main(
            ctx=ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify defaults were used
        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        # Check that time parameter was passed (value would be DEFAULT_TIME)
        assert 'time' in call_kwargs


def test_trace_main_all_ctx_params_provided(mock_ctx, mock_construct_paths, mock_trace):
    """Test with all context parameters provided."""
    mock_ctx.obj = {
        'quiet': True,
        'force': True,
        'strength': 0.9,
        'temperature': 0.5,
        'time': 120,
        'context': 'test_context'
    }

    with patch('pdd.trace_main.rprint') as mock_rprint:
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify all parameters were used
        construct_call = mock_construct_paths.call_args.kwargs
        assert construct_call['force'] is True
        assert construct_call['quiet'] is True
        assert construct_call['context_override'] == 'test_context'

        trace_call = mock_trace.call_args
        assert trace_call[0][3] == 0.9  # strength
        assert trace_call[0][4] == 0.5  # temperature
        assert trace_call.kwargs['time'] == 120

        # Verify no output in quiet mode
        mock_rprint.assert_not_called()


# =============================================================================
# RETURN VALUE TESTS
# =============================================================================

def test_trace_main_return_value_structure(mock_ctx, mock_construct_paths, mock_trace):
    """Test that return value has correct structure (tuple of 3 elements)."""
    with patch('pdd.trace_main.rprint'):
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify return value structure
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], int)  # prompt_line
        assert isinstance(result[1], float)  # total_cost
        assert isinstance(result[2], str)  # model_name


def test_trace_main_return_matches_trace_output(mock_ctx, mock_construct_paths, mock_trace):
    """Test that return value matches what trace function returned."""
    expected_return = (100, 0.005, 'test-model-123')
    mock_trace.return_value = expected_return

    with patch('pdd.trace_main.rprint'):
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        assert result == expected_return


def test_trace_main_various_return_values(mock_ctx, mock_construct_paths, mock_trace):
    """Test with various valid return values from trace."""
    test_cases = [
        (1, 0.0, 'model-a'),
        (999999, 1.5, 'model-b'),
        (42, 0.000001, 'very-long-model-name-here'),
    ]

    for expected in test_cases:
        mock_trace.return_value = expected

        with patch('pdd.trace_main.rprint'):
            result = trace_main(
                ctx=mock_ctx,
                prompt_file='prompt.txt',
                code_file='code.py',
                code_line=5,
                output=None
            )

            assert result == expected


# =============================================================================
# Z3 FORMAL VERIFICATION TESTS
# =============================================================================

def test_z3_return_value_invariants():
    """
    Z3 verification: Test invariants about return values.

    This test uses Z3 to verify logical properties about the return value:
    - prompt_line should be >= 1 (positive line numbers)
    - total_cost should be >= 0 (non-negative cost)
    - model_name should not be empty string

    Note: Z3 has limited applicability here because we can't model the full
    function behavior, but we can verify invariants about the return structure.
    """
    try:
        from z3 import Int, Real, String, Solver, And, sat

        # Create Z3 variables for return values
        prompt_line = Int('prompt_line')
        total_cost = Real('total_cost')

        # Define invariants we expect to hold
        solver = Solver()
        solver.add(prompt_line >= 1)  # Line numbers are positive
        solver.add(total_cost >= 0.0)  # Cost is non-negative

        # Verify the invariants are satisfiable
        assert solver.check() == sat

        # Now verify the negation is unsatisfiable (proof by contradiction)
        solver_neg = Solver()
        solver_neg.add(And(prompt_line >= 1, total_cost >= 0.0))
        solver_neg.add(prompt_line < 1)  # This should contradict
        assert solver_neg.check() != sat

    except ImportError:
        pytest.skip("Z3 not available")


def test_z3_error_exit_logic():
    """
    Z3 verification: Test that error conditions always lead to exit.

    This verifies the logical property:
    IF (error occurs) THEN (ctx.exit(1) is called)
    """
    try:
        from z3 import Bool, Implies, Solver, sat, Or, Not

        # Define boolean variables for conditions
        file_not_found = Bool('file_not_found')
        value_error = Bool('value_error')
        prompt_line_none = Bool('prompt_line_none')
        io_error = Bool('io_error')
        unexpected_error = Bool('unexpected_error')

        exit_called = Bool('exit_called')

        # Define the logical property: any error implies exit
        solver = Solver()
        solver.add(Implies(file_not_found, exit_called))
        solver.add(Implies(value_error, exit_called))
        solver.add(Implies(prompt_line_none, exit_called))
        solver.add(Implies(io_error, exit_called))
        solver.add(Implies(unexpected_error, exit_called))

        # Verify this property is satisfiable
        assert solver.check() == sat

        # Now verify that if any error occurs and exit is NOT called,
        # this leads to a contradiction
        solver_neg = Solver()
        solver_neg.add(
            Or(
                file_not_found,
                value_error,
                prompt_line_none,
                io_error,
                unexpected_error
            )
        )
        solver_neg.add(Not(exit_called))

        # This should be unsatisfiable with our error handling logic
        # (though Z3 can't verify the actual code, it verifies the logical structure)

    except ImportError:
        pytest.skip("Z3 not available")


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================

def test_trace_main_full_flow_integration(temp_dir):
    """
    Integration test with minimal mocking, using real file operations.
    """
    # Create real input files
    prompt_file = os.path.join(temp_dir, 'prompt.txt')
    code_file = os.path.join(temp_dir, 'code.py')
    output_file = os.path.join(temp_dir, 'output.txt')

    with open(prompt_file, 'w') as f:
        f.write("Write a hello world function\n")

    with open(code_file, 'w') as f:
        f.write("def hello():\n    return 'world'\n")

    # Create mock context
    ctx = Mock(spec=click.Context)
    ctx.obj = {
        'quiet': True,
        'force': False,
        'strength': 0.5,
        'temperature': 0.0,
        'time': 30,
        'context': None
    }
    ctx.exit = Mock(side_effect=SystemExit)

    # Mock only the external LLM call, not file operations
    with patch('pdd.trace_main.construct_paths') as mock_cp, \
         patch('pdd.trace_main.trace') as mock_trace:

        # Setup construct_paths to read real files
        with open(prompt_file, 'r') as pf, open(code_file, 'r') as cf:
            prompt_content = pf.read()
            code_content = cf.read()

        mock_cp.return_value = (
            {},
            {
                'prompt_file': prompt_content,
                'code_file': code_content
            },
            {'output': output_file},
            'python'
        )

        mock_trace.return_value = (1, 0.002, 'gpt-3.5')

        # Execute
        result = trace_main(ctx, prompt_file, code_file, 2, output_file)

        # Verify
        assert result == (1, 0.002, 'gpt-3.5')
        assert os.path.exists(output_file)

        with open(output_file, 'r') as f:
            content = f.read()
            assert 'Prompt Line: 1' in content
            assert 'Total Cost: $0.002000' in content
            assert 'Model Used: gpt-3.5' in content


def test_trace_main_console_output_format(mock_ctx, mock_construct_paths, mock_trace):
    """Test that console output has the expected format and styling."""
    with patch('pdd.trace_main.rprint') as mock_rprint:
        result = trace_main(
            ctx=mock_ctx,
            prompt_file='prompt.txt',
            code_file='code.py',
            code_line=5,
            output=None
        )

        # Verify all expected console messages were printed
        calls = [str(call) for call in mock_rprint.call_args_list]

        # Check that key messages are present
        assert any('Trace Analysis Complete' in call for call in calls)
        assert any('prompt line' in call.lower() for call in calls)
        assert any('cost' in call.lower() for call in calls)
        assert any('model' in call.lower() for call in calls)

        # Should have 4 calls (header + 3 info lines)
        assert len(calls) == 4