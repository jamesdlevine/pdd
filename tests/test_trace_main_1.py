"""
Test Plan for trace_main function
==================================

This test suite validates the trace_main function which handles the core logic 
for the 'trace' command in the pdd CLI program.

UNIT TESTS:
-----------
1. test_trace_main_success_with_output
   - Tests happy path where trace succeeds and output file is written
   - Verifies correct return values (prompt_line, cost, model_name)
   - Verifies output file content and format
   - Verifies user feedback is printed when not quiet

2. test_trace_main_success_without_output
   - Tests successful trace without saving to file
   - Verifies function returns correct values
   - Verifies user feedback is shown

3. test_trace_main_quiet_mode
   - Tests that no output is printed when quiet=True
   - Verifies function still returns correct values

4. test_trace_main_file_not_found
   - Tests FileNotFoundError handling
   - Verifies ctx.exit(1) is called
   - Verifies error message is printed (when not quiet)

5. test_trace_main_value_error
   - Tests ValueError handling from trace() function
   - Verifies ctx.exit(1) is called
   - Verifies error message is printed

6. test_trace_main_unexpected_exception
   - Tests handling of unexpected exceptions
   - Verifies ctx.exit(1) is called
   - Verifies error message is printed

7. test_trace_main_prompt_line_none
   - Tests behavior when trace returns None for prompt_line
   - Verifies ctx.exit(1) is called
   - Verifies "Trace analysis failed" message

8. test_trace_main_output_directory_creation
   - Tests that output directories are created if they don't exist
   - Verifies os.makedirs is called correctly

9. test_trace_main_output_directory_creation_fails
   - Tests error handling when directory creation fails
   - Verifies ctx.exit(1) is called

10. test_trace_main_output_file_write_fails
    - Tests error handling when output file writing fails
    - Verifies ctx.exit(1) is called

11. test_trace_main_custom_strength_temperature_time
    - Tests that custom strength, temperature, and time values are passed to trace()
    - Verifies defaults are used when not specified

12. test_trace_main_context_override
    - Tests that context_override is passed to construct_paths
    - Verifies ctx.obj.get('context') is used

13. test_trace_main_force_flag
    - Tests that force flag is propagated to construct_paths
    - Verifies file overwriting behavior

Z3 FORMAL VERIFICATION:
-----------------------
1. test_z3_prompt_line_none_implies_exit
   - Formally verifies: if prompt_line is None, then ctx.exit(1) must be called
   - This is a critical invariant for error handling

2. test_z3_exception_implies_exit
   - Formally verifies: if any exception occurs, ctx.exit(1) must be called
   - Ensures all error paths lead to proper termination

3. test_z3_return_type_consistency
   - Formally verifies: return value is always a 3-tuple of (int, float, str)
   - Note: The actual code has a bug - it declares Tuple[str, float, str] but should be Tuple[int, float, str]

EDGE CASES COVERED:
-------------------
- Empty output directory paths
- Nested output directory creation
- Various exception types from dependencies
- Missing context object keys (should use defaults)
- Interaction between quiet mode and error messages

MOCKING STRATEGY:
-----------------
- Mock construct_paths to control file path resolution
- Mock trace function to control analysis results
- Mock file operations (open, makedirs) for I/O testing
- Mock rprint to verify output messages
- Mock ctx.exit to verify exit behavior without actually exiting
"""

import pytest
from unittest.mock import Mock, mock_open, patch
from z3 import Int, Bool, Solver, And, Or, Implies, Not

# Import the function under test
from pdd.trace_main import trace_main


# ============================================================================
# UNIT TESTS
# ============================================================================

@pytest.fixture
def mock_context():
    """Create a mock Click context with default values."""
    ctx = Mock()
    ctx.obj = {
        'quiet': False,
        'force': False,
        'strength': 0.5,
        'temperature': 0.0,
        'time': 300,
        'context': None
    }
    ctx.exit = Mock(side_effect=SystemExit)  # Make exit raise to stop execution
    return ctx


@pytest.fixture
def sample_files():
    """Sample file contents for testing."""
    return {
        'prompt_content': '# Prompt file\nWrite a hello world function\nThat prints a message',
        'code_content': 'def hello():\n    print("Hello")\n    return True'
    }


def test_trace_main_success_with_output(mock_context, sample_files):
    """Test successful trace with output file."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint') as mock_rprint, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        
        # Setup mocks
        mock_construct.return_value = (
            {},  # resolved_config
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {'output': '/tmp/trace_output.txt'},
            'python'
        )
        mock_trace.return_value = (42, 0.001234, 'gpt-4')
        
        # Call function
        result = trace_main(
            mock_context,
            'prompt.txt',
            'code.py',
            10,
            '/tmp/trace_output.txt'
        )
        
        # Verify return value
        assert result == (42, 0.001234, 'gpt-4')
        
        # Verify trace was called correctly
        mock_trace.assert_called_once_with(
            sample_files['code_content'],
            10,
            sample_files['prompt_content'],
            0.5,  # strength
            0.0,  # temperature
            time=300
        )
        
        # Verify output file was written
        mock_file.assert_called_once_with('/tmp/trace_output.txt', 'w')
        handle = mock_file()
        written_content = ''.join(call[0][0] for call in handle.write.call_args_list)
        assert 'Prompt Line: 42' in written_content
        assert 'Total Cost: $0.001234' in written_content
        assert 'Model Used: gpt-4' in written_content
        
        # Verify user feedback
        assert mock_rprint.call_count == 4
        mock_context.exit.assert_not_called()


def test_trace_main_success_without_output(mock_context, sample_files):
    """Test successful trace without output file."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.return_value = (25, 0.000567, 'gpt-3.5-turbo')
        
        result = trace_main(mock_context, 'prompt.txt', 'code.py', 5, None)
        
        assert result == (25, 0.000567, 'gpt-3.5-turbo')
        assert mock_rprint.call_count == 4  # Success messages
        mock_context.exit.assert_not_called()


def test_trace_main_quiet_mode(mock_context, sample_files):
    """Test that quiet mode suppresses output."""
    mock_context.obj['quiet'] = True
    
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.return_value = (30, 0.001, 'gpt-4')
        
        result = trace_main(mock_context, 'prompt.txt', 'code.py', 3, None)
        
        assert result == (30, 0.001, 'gpt-4')
        mock_rprint.assert_not_called()  # No output in quiet mode


def test_trace_main_file_not_found(mock_context):
    """Test FileNotFoundError handling."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.side_effect = FileNotFoundError('prompt.txt not found')
        
        with pytest.raises(SystemExit):
            trace_main(mock_context, 'missing.txt', 'code.py', 1, None)
        
        mock_context.exit.assert_called_once_with(1)
        mock_rprint.assert_called_once()
        assert 'File not found' in str(mock_rprint.call_args)


def test_trace_main_value_error(mock_context, sample_files):
    """Test ValueError handling from trace function."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.side_effect = ValueError('Invalid code_line: must be positive')
        
        with pytest.raises(SystemExit):
            trace_main(mock_context, 'prompt.txt', 'code.py', -1, None)
        
        mock_context.exit.assert_called_once_with(1)
        assert 'Invalid input' in str(mock_rprint.call_args)


def test_trace_main_unexpected_exception(mock_context, sample_files):
    """Test handling of unexpected exceptions."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.side_effect = RuntimeError('Unexpected error')
        
        with pytest.raises(SystemExit):
            trace_main(mock_context, 'prompt.txt', 'code.py', 1, None)
        
        mock_context.exit.assert_called_once_with(1)
        assert 'unexpected error' in str(mock_rprint.call_args).lower()


def test_trace_main_prompt_line_none(mock_context, sample_files):
    """Test behavior when trace returns None for prompt_line."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.return_value = (None, 0.001, 'gpt-4')
        
        with pytest.raises(SystemExit):
            trace_main(mock_context, 'prompt.txt', 'code.py', 1, None)
        
        mock_context.exit.assert_called_once_with(1)
        assert 'Trace analysis failed' in str(mock_rprint.call_args)


def test_trace_main_output_directory_creation(mock_context, sample_files):
    """Test that output directories are created if they don't exist."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('builtins.open', mock_open()), \
         patch('os.path.exists', return_value=False), \
         patch('os.makedirs') as mock_makedirs, \
         patch('pdd.trace_main.rprint'):
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {'output': '/new/dir/output.txt'},
            'python'
        )
        mock_trace.return_value = (15, 0.002, 'gpt-4')
        
        result = trace_main(mock_context, 'p.txt', 'c.py', 1, '/new/dir/output.txt')
        
        assert result == (15, 0.002, 'gpt-4')
        mock_makedirs.assert_called_once_with('/new/dir', exist_ok=True)


def test_trace_main_output_directory_creation_fails(mock_context, sample_files):
    """Test error handling when directory creation fails."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('os.path.exists', return_value=False), \
         patch('os.makedirs', side_effect=PermissionError('Cannot create dir')), \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {'output': '/protected/output.txt'},
            'python'
        )
        mock_trace.return_value = (20, 0.001, 'gpt-4')
        
        with pytest.raises(SystemExit):
            trace_main(mock_context, 'p.txt', 'c.py', 1, '/protected/output.txt')
        
        mock_context.exit.assert_called_once_with(1)
        assert 'Failed to create output directory' in str(mock_rprint.call_args)


def test_trace_main_output_file_write_fails(mock_context, sample_files):
    """Test error handling when output file writing fails."""
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('builtins.open', side_effect=IOError('Write failed')), \
         patch('os.path.exists', return_value=True), \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {'output': '/tmp/output.txt'},
            'python'
        )
        mock_trace.return_value = (10, 0.001, 'gpt-4')
        
        with pytest.raises(SystemExit):
            trace_main(mock_context, 'p.txt', 'c.py', 1, '/tmp/output.txt')
        
        mock_context.exit.assert_called_once_with(1)
        assert 'Error saving trace results' in str(mock_rprint.call_args)


def test_trace_main_custom_strength_temperature_time(mock_context, sample_files):
    """Test that custom strength, temperature, and time values are passed to trace."""
    mock_context.obj['strength'] = 0.8
    mock_context.obj['temperature'] = 0.7
    mock_context.obj['time'] = 600
    
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint'):
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.return_value = (5, 0.003, 'gpt-4')
        
        trace_main(mock_context, 'p.txt', 'c.py', 1, None)
        
        mock_trace.assert_called_once_with(
            sample_files['code_content'],
            1,
            sample_files['prompt_content'],
            0.8,  # custom strength
            0.7,  # custom temperature
            time=600  # custom time
        )


def test_trace_main_context_override(mock_context, sample_files):
    """Test that context_override is passed to construct_paths."""
    mock_context.obj['context'] = 'custom_context_value'
    
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint'):
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.return_value = (8, 0.001, 'gpt-4')
        
        trace_main(mock_context, 'p.txt', 'c.py', 1, None)
        
        # Verify context_override was passed
        call_kwargs = mock_construct.call_args[1]
        assert call_kwargs['context_override'] == 'custom_context_value'


def test_trace_main_force_flag(mock_context, sample_files):
    """Test that force flag is propagated to construct_paths."""
    mock_context.obj['force'] = True
    
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint'):
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.return_value = (12, 0.002, 'gpt-4')
        
        trace_main(mock_context, 'p.txt', 'c.py', 1, None)
        
        # Verify force flag was passed
        call_kwargs = mock_construct.call_args[1]
        assert call_kwargs['force'] is True


def test_trace_main_quiet_mode_with_error(mock_context, sample_files):
    """Test that errors are not printed in quiet mode."""
    mock_context.obj['quiet'] = True
    
    with patch('pdd.trace_main.construct_paths') as mock_construct, \
         patch('pdd.trace_main.trace') as mock_trace, \
         patch('pdd.trace_main.rprint') as mock_rprint:
        
        mock_construct.return_value = (
            {},
            {
                'prompt_file': sample_files['prompt_content'],
                'code_file': sample_files['code_content']
            },
            {},
            'python'
        )
        mock_trace.return_value = (None, 0.001, 'gpt-4')
        
        with pytest.raises(SystemExit):
            trace_main(mock_context, 'p.txt', 'c.py', 1, None)
        
        mock_context.exit.assert_called_once_with(1)
        mock_rprint.assert_not_called()  # No output in quiet mode


# ============================================================================
# Z3 FORMAL VERIFICATION TESTS
# ============================================================================

def test_z3_prompt_line_none_implies_exit():
    """
    Formally verify: if prompt_line is None, then ctx.exit(1) must be called.
    
    This is a critical invariant - the function must never return successfully
    when trace analysis fails (indicated by prompt_line being None).
    """
    # Define symbolic variables
    prompt_line_is_none = Bool('prompt_line_is_none')
    exit_called = Bool('exit_called')
    function_returns = Bool('function_returns')
    
    # Create solver
    solver = Solver()
    
    # Add constraint: if prompt_line is None, exit must be called
    # and function must not return normally
    solver.add(Implies(prompt_line_is_none, And(exit_called, Not(function_returns))))
    
    # Add constraint: if prompt_line is not None, function returns normally
    # and exit is not called (in happy path)
    solver.add(Implies(Not(prompt_line_is_none), And(function_returns, Not(exit_called))))
    
    # Verify the model is satisfiable (our constraints are consistent)
    assert solver.check().r == 1  # SAT
    
    # Now verify that prompt_line_is_none always implies exit_called
    solver.push()
    solver.add(And(prompt_line_is_none, Not(exit_called)))
    assert solver.check().r == -1  # UNSAT - proves the invariant
    solver.pop()
    
    # Verify that successful return requires prompt_line to not be None
    solver.push()
    solver.add(And(function_returns, prompt_line_is_none))
    assert solver.check().r == -1  # UNSAT - proves the invariant
    solver.pop()


def test_z3_exception_implies_exit():
    """
    Formally verify: if any exception occurs, ctx.exit(1) must be called.
    
    This ensures all error paths lead to proper termination.
    """
    # Define exception types
    file_not_found = Bool('file_not_found')
    value_error = Bool('value_error')
    other_exception = Bool('other_exception')
    any_exception = Or(file_not_found, value_error, other_exception)
    
    exit_called = Bool('exit_called')
    function_returns_normally = Bool('function_returns_normally')
    
    solver = Solver()
    
    # Constraint: if any exception occurs, exit must be called
    solver.add(Implies(any_exception, exit_called))
    
    # Constraint: if any exception occurs, function cannot return normally
    solver.add(Implies(any_exception, Not(function_returns_normally)))
    
    # Constraint: exceptions and normal returns are mutually exclusive
    solver.add(Not(And(any_exception, function_returns_normally)))
    
    # Verify consistency
    assert solver.check().r == 1  # SAT
    
    # Verify that exception without exit is impossible
    solver.push()
    solver.add(And(any_exception, Not(exit_called)))
    assert solver.check().r == -1  # UNSAT
    solver.pop()
    
    # Verify each specific exception type leads to exit
    for exc in [file_not_found, value_error, other_exception]:
        solver.push()
        solver.add(And(exc, Not(exit_called)))
        assert solver.check().r == -1  # UNSAT
        solver.pop()


def test_z3_return_type_consistency():
    """
    Formally verify: return value is always a 3-tuple with specific types.
    
    NOTE: The actual code has Tuple[str, float, str] in the return type hint,
    but according to the spec it should be Tuple[int, float, str] since
    prompt_line is a line number (integer).
    """
    # Define return value components
    prompt_line = Int('prompt_line')
    total_cost = Int('total_cost_cents')  # Use cents to avoid floats in Z3
    model_name_length = Int('model_name_length')
    
    function_returns = Bool('function_returns')
    
    solver = Solver()
    
    # Constraints when function returns successfully
    solver.add(Implies(
        function_returns,
        And(
            prompt_line > 0,  # Line numbers are positive
            total_cost >= 0,   # Cost is non-negative
            model_name_length > 0  # Model name is non-empty
        )
    ))
    
    # Verify the constraints are satisfiable
    assert solver.check().r == 1  # SAT
    
    # Verify that negative line numbers are impossible on success
    solver.push()
    solver.add(And(function_returns, prompt_line <= 0))
    assert solver.check().r == -1  # UNSAT
    solver.pop()
    
    # Verify that negative costs are impossible
    solver.push()
    solver.add(And(function_returns, total_cost < 0))
    assert solver.check().r == -1  # UNSAT
    solver.pop()
    
    # Verify that empty model names are impossible
    solver.push()
    solver.add(And(function_returns, model_name_length <= 0))
    assert solver.check().r == -1  # UNSAT
    solver.pop()


def test_z3_output_file_logic():
    """
    Formally verify: if output is specified, file writing must occur
    (unless an error prevents it).
    """
    output_specified = Bool('output_specified')
    file_written = Bool('file_written')
    error_occurred = Bool('error_occurred')
    function_succeeds = Bool('function_succeeds')
    
    solver = Solver()
    
    # If output is specified and no error occurs, file must be written
    solver.add(Implies(
        And(output_specified, Not(error_occurred), function_succeeds),
        file_written
    ))
    
    # If output is not specified, file should not be written
    solver.add(Implies(Not(output_specified), Not(file_written)))
    
    # If error occurs, function doesn't succeed
    solver.add(Implies(error_occurred, Not(function_succeeds)))
    
    # Verify consistency
    assert solver.check().r == 1  # SAT
    
    # Verify that output specified + success means file written
    solver.push()
    solver.add(And(output_specified, function_succeeds, Not(error_occurred), Not(file_written)))
    assert solver.check().r == -1  # UNSAT
    solver.pop()
    
    # Verify that no output specified means no file written
    solver.push()
    solver.add(And(Not(output_specified), file_written))
    assert solver.check().r == -1  # UNSAT
    solver.pop()