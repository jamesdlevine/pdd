"""
Comprehensive test suite for auto_deps_main function.

TEST PLAN:
==========

This test suite uses unit tests rather than Z3 formal verification because:
- The function primarily orchestrates I/O operations and external dependencies
- It has side effects (file writing, console output)
- The logic is straightforward orchestration rather than complex algorithms

Test Categories:
----------------

1. SUCCESSFUL EXECUTION TESTS
   - Test normal flow with all parameters provided
   - Test with minimal parameters (optional ones as None)
   - Verify construct_paths is called with correct arguments
   - Verify insert_includes is called with correct arguments
   - Verify output files are written correctly
   - Verify return values are correct

2. FORCE SCAN TESTS
   - Test force_scan=True with existing CSV file (should delete)
   - Test force_scan=True with non-existent CSV file (should not error)
   - Test force_scan=False with existing CSV file (should not delete)
   - Verify warning message is printed when deleting

3. OUTPUT MODE TESTS (QUIET/VERBOSE)
   - Test quiet mode (no console output)
   - Test verbose mode (console output present)
   - Verify success messages in verbose mode
   - Verify cost and model information displayed

4. CSV OUTPUT TESTS
   - Test with non-empty CSV output (should save file)
   - Test with empty CSV output (should not save file)
   - Test with None CSV output (should not save file)

5. CONTEXT PARAMETER TESTS
   - Test extraction of strength parameter (with default)
   - Test extraction of temperature parameter (with default)
   - Test extraction of time parameter (with default)
   - Test extraction of force parameter
   - Test extraction of quiet parameter
   - Test extraction of context_override parameter

6. EXCEPTION HANDLING TESTS
   - Test click.Abort exception (should re-raise)
   - Test general exception (should return error tuple)
   - Test exception in quiet mode (should print error)
   - Test exception in verbose mode (should print error)
   - Verify error tuple format (empty string, 0.0, error message)

7. PROGRESS CALLBACK TESTS
   - Test that progress_callback is passed to insert_includes
   - Test with None progress_callback

8. FILE PATH RESOLUTION TESTS
   - Test CSV path from output_file_paths
   - Test default CSV path when not in output_file_paths
   - Test output path resolution from construct_paths

Edge Cases:
-----------
- CSV file path contains special characters
- Output directory doesn't exist (Path.write_text should handle)
- Context object has partial parameters
- Context object is missing expected keys
- insert_includes returns empty strings
- Very long file paths
- Unicode in file paths and content

Mocking Strategy:
-----------------
- Mock construct_paths to control path resolution
- Mock insert_includes to control return values
- Mock Path operations for file I/O
- Mock rprint for output verification
- Use pytest fixtures for reusable test data
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import click

# Import the function under test
from pdd.auto_deps_main import auto_deps_main


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_ctx():
    """Create a mock Click context with default configuration."""
    ctx = Mock(spec=click.Context)
    ctx.obj = {
        'force': False,
        'quiet': False,
        'strength': 0.5,
        'temperature': 0,
        'time': 0.25,
        'context': None,
        'confirm_callback': None
    }
    return ctx


@pytest.fixture
def mock_ctx_quiet():
    """Create a mock Click context with quiet mode enabled."""
    ctx = Mock(spec=click.Context)
    ctx.obj = {
        'force': False,
        'quiet': True,
        'strength': 0.5,
        'temperature': 0,
        'time': 0.25,
        'context': None,
        'confirm_callback': None
    }
    return ctx


@pytest.fixture
def mock_ctx_minimal():
    """Create a mock Click context with minimal configuration (testing defaults)."""
    ctx = Mock(spec=click.Context)
    ctx.obj = {}  # Empty context to test defaults
    return ctx


@pytest.fixture
def sample_construct_paths_return():
    """Sample return value for construct_paths."""
    return (
        {'some': 'config'},  # resolved_config
        {'prompt_file': 'Sample prompt content here'},  # input_strings
        {
            'output': '/tmp/output_prompt.prompt',
            'csv': '/tmp/project_deps.csv'
        },  # output_file_paths
        'python'  # language
    )


@pytest.fixture
def sample_insert_includes_return():
    """Sample return value for insert_includes."""
    return (
        'Modified prompt with dependencies',  # modified_prompt
        'file1.py,summary1\nfile2.py,summary2',  # csv_output
        0.0042,  # total_cost
        'gpt-4'  # model_name
    )


# ============================================================================
# TEST 1: SUCCESSFUL EXECUTION TESTS
# ============================================================================

def test_successful_execution_with_all_parameters(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return,
    tmp_path
):
    """Test successful execution with all parameters provided."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint') as mock_rprint:

        # Setup mocks
        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return

        # Mock Path for file operations
        mock_output_path = MagicMock()
        mock_csv_path = MagicMock()
        mock_path.return_value = mock_output_path

        def path_side_effect(arg):
            if 'output_prompt' in str(arg):
                return mock_output_path
            elif 'project_deps' in str(arg):
                return mock_csv_path
            return MagicMock()

        mock_path.side_effect = path_side_effect

        # Execute
        result = auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=False
        )

        # Verify construct_paths was called correctly
        mock_construct.assert_called_once()
        call_kwargs = mock_construct.call_args[1]
        assert call_kwargs['input_file_paths'] == {'prompt_file': 'test.prompt'}
        assert call_kwargs['command'] == 'auto-deps'
        assert call_kwargs['command_options'] == {'output': 'output.prompt', 'csv': 'deps.csv'}

        # Verify insert_includes was called correctly
        mock_insert.assert_called_once()
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['input_prompt'] == 'Sample prompt content here'
        assert insert_kwargs['directory_path'] == 'examples/'
        assert insert_kwargs['csv_filename'] == '/tmp/project_deps.csv'
        assert insert_kwargs['strength'] == 0.5
        assert insert_kwargs['temperature'] == 0
        assert insert_kwargs['time'] == 0.25
        assert insert_kwargs['verbose'] is True

        # Verify return value
        assert result == ('Modified prompt with dependencies', 0.0042, 'gpt-4')

        # Verify success messages were printed
        assert mock_rprint.call_count >= 5


def test_successful_execution_with_minimal_parameters(
    mock_ctx_minimal,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test successful execution with minimal parameters (testing defaults)."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'), \
         patch('pdd.auto_deps_main.DEFAULT_STRENGTH', 0.5), \
         patch('pdd.auto_deps_main.DEFAULT_TIME', 0.25):

        # Setup mocks
        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Execute with minimal parameters
        result = auto_deps_main(
            ctx=mock_ctx_minimal,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify insert_includes received default values
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['strength'] == 0.5  # DEFAULT_STRENGTH
        assert insert_kwargs['temperature'] == 0
        assert insert_kwargs['time'] == 0.25  # DEFAULT_TIME

        # Verify return value
        assert result[0] == 'Modified prompt with dependencies'
        assert result[1] == 0.0042
        assert result[2] == 'gpt-4'


def test_output_files_written_correctly(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test that output files are written with correct content."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return

        # Mock Path instances
        mock_output_path_obj = MagicMock()
        mock_csv_path_obj = MagicMock()

        path_instances = {}

        def path_constructor(path_str):
            if path_str not in path_instances:
                if 'output_prompt' in str(path_str):
                    path_instances[path_str] = mock_output_path_obj
                elif 'project_deps' in str(path_str):
                    path_instances[path_str] = mock_csv_path_obj
                else:
                    path_instances[path_str] = MagicMock()
            return path_instances[path_str]

        mock_path.side_effect = path_constructor
        mock_csv_path_obj.exists.return_value = False

        # Execute
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=False
        )

        # Verify output prompt was written
        mock_output_path_obj.write_text.assert_called_once_with(
            'Modified prompt with dependencies',
            encoding='utf-8'
        )

        # Verify CSV was written
        mock_csv_path_obj.write_text.assert_called_once_with(
            'file1.py,summary1\nfile2.py,summary2',
            encoding='utf-8'
        )


# ============================================================================
# TEST 2: FORCE SCAN TESTS
# ============================================================================

def test_force_scan_with_existing_csv(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test force_scan=True with existing CSV file (should delete)."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint') as mock_rprint:

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return

        # Mock CSV path to exist
        mock_csv_path_obj = MagicMock()
        mock_csv_path_obj.exists.return_value = True
        mock_output_path_obj = MagicMock()

        def path_constructor(path_str):
            if 'project_deps' in str(path_str):
                return mock_csv_path_obj
            elif 'output_prompt' in str(path_str):
                return mock_output_path_obj
            return MagicMock()

        mock_path.side_effect = path_constructor

        # Execute with force_scan=True
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=True
        )

        # Verify CSV file was deleted
        mock_csv_path_obj.unlink.assert_called_once()

        # Verify warning message was printed
        warning_calls = [call for call in mock_rprint.call_args_list
                        if 'Removing existing CSV file' in str(call)]
        assert len(warning_calls) > 0


def test_force_scan_with_nonexistent_csv(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test force_scan=True with non-existent CSV file (should not error)."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return

        # Mock CSV path to not exist
        mock_csv_path_obj = MagicMock()
        mock_csv_path_obj.exists.return_value = False
        mock_output_path_obj = MagicMock()

        def path_constructor(path_str):
            if 'project_deps' in str(path_str):
                return mock_csv_path_obj
            elif 'output_prompt' in str(path_str):
                return mock_output_path_obj
            return MagicMock()

        mock_path.side_effect = path_constructor

        # Execute with force_scan=True - should not raise error
        result = auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=True
        )

        # Verify unlink was NOT called
        mock_csv_path_obj.unlink.assert_not_called()

        # Verify execution succeeded
        assert result[2] == 'gpt-4'


def test_force_scan_false_preserves_csv(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test force_scan=False with existing CSV file (should not delete)."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return

        # Mock CSV path to exist
        mock_csv_path_obj = MagicMock()
        mock_csv_path_obj.exists.return_value = True
        mock_output_path_obj = MagicMock()

        def path_constructor(path_str):
            if 'project_deps' in str(path_str):
                return mock_csv_path_obj
            elif 'output_prompt' in str(path_str):
                return mock_output_path_obj
            return MagicMock()

        mock_path.side_effect = path_constructor

        # Execute with force_scan=False
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=False
        )

        # Verify unlink was NOT called
        mock_csv_path_obj.unlink.assert_not_called()


# ============================================================================
# TEST 3: OUTPUT MODE TESTS (QUIET/VERBOSE)
# ============================================================================

def test_quiet_mode_no_output(
    mock_ctx_quiet,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test quiet mode produces no console output."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint') as mock_rprint:

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Execute in quiet mode
        auto_deps_main(
            ctx=mock_ctx_quiet,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=False
        )

        # Verify no output was printed
        mock_rprint.assert_not_called()

        # Verify verbose parameter was False
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['verbose'] is False


def test_verbose_mode_with_output(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test verbose mode produces console output."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint') as mock_rprint:

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Execute in verbose mode (quiet=False)
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=False
        )

        # Verify output was printed (multiple success messages)
        assert mock_rprint.call_count >= 5

        # Verify specific messages were printed
        printed_messages = [str(call) for call in mock_rprint.call_args_list]
        assert any('Successfully analyzed' in msg for msg in printed_messages)
        assert any('Model used' in msg for msg in printed_messages)
        assert any('Total cost' in msg for msg in printed_messages)


# ============================================================================
# TEST 4: CSV OUTPUT TESTS
# ============================================================================

def test_csv_saved_with_nonempty_output(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test CSV file is saved when csv_output is non-empty."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return

        mock_csv_path_obj = MagicMock()
        mock_output_path_obj = MagicMock()

        def path_constructor(path_str):
            if 'project_deps' in str(path_str):
                return mock_csv_path_obj
            elif 'output_prompt' in str(path_str):
                return mock_output_path_obj
            return MagicMock()

        mock_path.side_effect = path_constructor
        mock_csv_path_obj.exists.return_value = False

        # Execute
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=False
        )

        # Verify CSV was written
        mock_csv_path_obj.write_text.assert_called_once()


def test_csv_not_saved_with_empty_output(
    mock_ctx,
    sample_construct_paths_return
):
    """Test CSV file is not saved when csv_output is empty."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        # Return empty CSV output
        mock_insert.return_value = (
            'Modified prompt',
            '',  # Empty CSV output
            0.001,
            'gpt-4'
        )

        mock_csv_path_obj = MagicMock()
        mock_output_path_obj = MagicMock()

        def path_constructor(path_str):
            if 'project_deps' in str(path_str):
                return mock_csv_path_obj
            elif 'output_prompt' in str(path_str):
                return mock_output_path_obj
            return MagicMock()

        mock_path.side_effect = path_constructor
        mock_csv_path_obj.exists.return_value = False

        # Execute
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path='deps.csv',
            output='output.prompt',
            force_scan=False
        )

        # Verify CSV was NOT written (empty string is falsy)
        mock_csv_path_obj.write_text.assert_not_called()


def test_csv_path_from_default(
    mock_ctx,
    sample_insert_includes_return
):
    """Test default CSV path when not in output_file_paths."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        # Return output_file_paths without 'csv' key
        mock_construct.return_value = (
            {},
            {'prompt_file': 'content'},
            {'output': '/tmp/output.prompt'},  # No 'csv' key
            'python'
        )
        mock_insert.return_value = sample_insert_includes_return

        mock_csv_path_obj = MagicMock()
        mock_output_path_obj = MagicMock()

        def path_constructor(path_str):
            if 'project_dependencies.csv' in str(path_str):
                return mock_csv_path_obj
            elif 'output.prompt' in str(path_str):
                return mock_output_path_obj
            return MagicMock()

        mock_path.side_effect = path_constructor
        mock_csv_path_obj.exists.return_value = False

        # Execute
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify insert_includes was called with default CSV filename
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['csv_filename'] == 'project_dependencies.csv'


# ============================================================================
# TEST 5: CONTEXT PARAMETER TESTS
# ============================================================================

def test_context_parameters_with_custom_values(mock_ctx):
    """Test extraction of custom context parameters."""
    # Set custom values in context
    mock_ctx.obj = {
        'force': True,
        'quiet': False,
        'strength': 0.8,
        'temperature': 0.5,
        'time': 0.75,
        'context': 'custom_context',
        'confirm_callback': lambda: True
    }

    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = (
            {},
            {'prompt_file': 'content'},
            {'output': '/tmp/out.prompt', 'csv': '/tmp/deps.csv'},
            'python'
        )
        mock_insert.return_value = ('modified', 'csv', 0.001, 'model')
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Execute
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify construct_paths received custom parameters
        construct_kwargs = mock_construct.call_args[1]
        assert construct_kwargs['force'] is True
        assert construct_kwargs['context_override'] == 'custom_context'

        # Verify insert_includes received custom parameters
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['strength'] == 0.8
        assert insert_kwargs['temperature'] == 0.5
        assert insert_kwargs['time'] == 0.75


def test_context_parameters_with_defaults(mock_ctx_minimal):
    """Test extraction of default context parameters when not specified."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'), \
         patch('pdd.auto_deps_main.DEFAULT_STRENGTH', 0.5), \
         patch('pdd.auto_deps_main.DEFAULT_TIME', 0.25):

        mock_construct.return_value = (
            {},
            {'prompt_file': 'content'},
            {'output': '/tmp/out.prompt', 'csv': '/tmp/deps.csv'},
            'python'
        )
        mock_insert.return_value = ('modified', 'csv', 0.001, 'model')
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Execute with minimal context
        auto_deps_main(
            ctx=mock_ctx_minimal,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify construct_paths received default parameters
        construct_kwargs = mock_construct.call_args[1]
        assert construct_kwargs['force'] is False
        assert construct_kwargs['quiet'] is False
        assert construct_kwargs['context_override'] is None

        # Verify insert_includes received default parameters
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['strength'] == 0.5  # DEFAULT_STRENGTH
        assert insert_kwargs['temperature'] == 0
        assert insert_kwargs['time'] == 0.25  # DEFAULT_TIME
        assert insert_kwargs['verbose'] is True  # not quiet


# ============================================================================
# TEST 6: EXCEPTION HANDLING TESTS
# ============================================================================

def test_click_abort_exception_reraised(mock_ctx):
    """Test that click.Abort exception is re-raised."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.rprint'):

        # Make construct_paths raise click.Abort
        mock_construct.side_effect = click.Abort()

        # Verify exception is re-raised
        with pytest.raises(click.Abort):
            auto_deps_main(
                ctx=mock_ctx,
                prompt_file='test.prompt',
                directory_path='examples/',
                auto_deps_csv_path=None,
                output=None,
                force_scan=False
            )


def test_general_exception_returns_error_tuple(mock_ctx):
    """Test that general exceptions return error tuple."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.rprint') as mock_rprint:

        # Make construct_paths raise a general exception
        mock_construct.side_effect = ValueError("Test error message")

        # Execute
        result = auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify error tuple format
        assert result[0] == ''
        assert result[1] == 0.0
        assert 'Error:' in result[2]
        assert 'Test error message' in result[2]

        # Verify error was printed (not in quiet mode)
        assert mock_rprint.called


def test_exception_in_quiet_mode(mock_ctx_quiet):
    """Test exception handling in quiet mode."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.rprint') as mock_rprint:

        mock_construct.side_effect = RuntimeError("Test error")

        # Execute
        result = auto_deps_main(
            ctx=mock_ctx_quiet,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify error tuple
        assert result[0] == ''
        assert result[1] == 0.0
        assert 'Error:' in result[2]

        # Verify no output in quiet mode
        mock_rprint.assert_not_called()


def test_exception_during_insert_includes(
    mock_ctx,
    sample_construct_paths_return
):
    """Test exception handling during insert_includes call."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path'), \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.side_effect = Exception("Insert includes failed")

        # Execute
        result = auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify error tuple
        assert result[0] == ''
        assert result[1] == 0.0
        assert 'Error:' in result[2]
        assert 'Insert includes failed' in result[2]


# ============================================================================
# TEST 7: PROGRESS CALLBACK TESTS
# ============================================================================

def test_progress_callback_passed_to_insert_includes(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test that progress_callback is passed to insert_includes."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Create a mock progress callback
        mock_progress = Mock()

        # Execute with progress callback
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False,
            progress_callback=mock_progress
        )

        # Verify progress_callback was passed to insert_includes
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['progress_callback'] == mock_progress


def test_none_progress_callback(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test with None progress_callback."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Execute with None progress callback (default)
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test.prompt',
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False,
            progress_callback=None
        )

        # Verify insert_includes was called with None callback
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['progress_callback'] is None


# ============================================================================
# TEST 8: FILE PATH RESOLUTION TESTS
# ============================================================================

def test_prompt_filename_passed_to_insert_includes(
    mock_ctx,
    sample_construct_paths_return,
    sample_insert_includes_return
):
    """Test that prompt_filename parameter is passed to insert_includes."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        mock_construct.return_value = sample_construct_paths_return
        mock_insert.return_value = sample_insert_includes_return
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        prompt_file = 'my_special_prompt.prompt'

        # Execute
        auto_deps_main(
            ctx=mock_ctx,
            prompt_file=prompt_file,
            directory_path='examples/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify prompt_filename was passed correctly
        insert_kwargs = mock_insert.call_args[1]
        assert insert_kwargs['prompt_filename'] == prompt_file


def test_special_characters_in_paths(
    mock_ctx,
    sample_insert_includes_return
):
    """Test handling of special characters in file paths."""
    with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
         patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
         patch('pdd.auto_deps_main.Path') as mock_path, \
         patch('pdd.auto_deps_main.rprint'):

        # Paths with special characters
        special_paths = (
            {},
            {'prompt_file': 'content'},
            {
                'output': '/tmp/output with spaces.prompt',
                'csv': '/tmp/deps-with-dashes.csv'
            },
            'python'
        )
        mock_construct.return_value = special_paths
        mock_insert.return_value = sample_insert_includes_return
        mock_path.return_value.write_text = MagicMock()
        mock_path.return_value.exists.return_value = False

        # Execute - should handle special characters correctly
        result = auto_deps_main(
            ctx=mock_ctx,
            prompt_file='test with spaces.prompt',
            directory_path='examples/sub dir/',
            auto_deps_csv_path=None,
            output=None,
            force_scan=False
        )

        # Verify execution succeeded
        assert result[2] == 'gpt-4'