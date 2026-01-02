"""
Comprehensive unit tests for auto_deps_main function.

TEST PLAN:
==========

This test suite verifies the auto_deps_main function which orchestrates the auto-deps
command workflow. The function integrates multiple components and handles various
scenarios.

METHODOLOGY:
-----------
Unit tests with mocking are used exclusively because:
- The function performs I/O operations (file reading/writing)
- It has side effects (deleting files, printing output) 
- It integrates multiple external components (construct_paths, insert_includes)
- The logic is procedural/orchestration rather than algorithmic
- Z3 formal verification is not suitable for testing I/O and integration logic

TEST CATEGORIES:
---------------

1. BASIC FUNCTIONALITY TESTS:
   - test_auto_deps_main_success: Verify successful execution with all components
   - test_auto_deps_main_with_csv_output: Verify CSV file is written when generated
   - test_auto_deps_main_without_csv_output: Verify CSV not written when empty

2. FORCE SCAN TESTS:
   - test_auto_deps_main_force_scan_deletes_existing_csv: Verify CSV deletion when exists
   - test_auto_deps_main_force_scan_no_csv_exists: Verify no error when CSV doesn't exist
   - test_auto_deps_main_no_force_scan_keeps_csv: Verify CSV not deleted without flag

3. QUIET MODE TESTS:
   - test_auto_deps_main_quiet_mode: Verify no output in quiet mode
   - test_auto_deps_main_non_quiet_mode: Verify output messages in normal mode

4. CONTEXT PARAMETER TESTS:
   - test_auto_deps_main_custom_strength_temperature: Verify custom LLM parameters
   - test_auto_deps_main_default_parameters: Verify default parameter values
   - test_auto_deps_main_with_time_budget: Verify time parameter passed correctly

5. PATH HANDLING TESTS:
   - test_auto_deps_main_custom_output_path: Verify custom output path used
   - test_auto_deps_main_custom_csv_path: Verify custom CSV path used
   - test_auto_deps_main_default_csv_path: Verify default CSV path fallback

6. EXCEPTION HANDLING TESTS:
   - test_auto_deps_main_click_abort_reraises: Verify click.Abort re-raised
   - test_auto_deps_main_general_exception_returns_error: Verify error tuple returned
   - test_auto_deps_main_exception_with_quiet_mode: Verify no error output in quiet

7. INTEGRATION TESTS:
   - test_auto_deps_main_construct_paths_called_correctly: Verify correct arguments
   - test_auto_deps_main_insert_includes_called_correctly: Verify correct arguments
   - test_auto_deps_main_progress_callback_passed: Verify callback passed through

8. EDGE CASES:
   - test_auto_deps_main_none_optional_parameters: Verify None handling
   - test_auto_deps_main_context_override: Verify context override passed
   - test_auto_deps_main_confirm_callback: Verify confirm_callback passed

COVERAGE GOALS:
--------------
- All code paths (normal, error, force_scan, quiet modes)
- All parameter combinations
- All exception handling paths
- All external function call patterns
- All file I/O operations
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import click

# Import the function under test from the actual module path
from pdd.auto_deps_main import auto_deps_main


class TestAutoDepMainSuccess:
    """Test successful execution scenarios."""
    
    def test_auto_deps_main_success(self) -> None:
        """Test successful execution with all components working."""
        # Setup
        ctx = Mock(spec=click.Context)
        ctx.obj = {
            'force': False,
            'quiet': False,
            'strength': 0.7,
            'temperature': 0.5,
            'time': 0.3,
            'context': None,
            'confirm_callback': None
        }
        
        prompt_file = "test_prompt.prompt"
        directory_path = "test_dir/"
        auto_deps_csv_path = "test_deps.csv"
        output = "test_output.prompt"
        force_scan = False
        
        # Mock construct_paths
        mock_resolved_config: dict = {}
        mock_input_strings = {"prompt_file": "test prompt content"}
        mock_output_file_paths = {
            "output": "output_path.prompt",
            "csv": "csv_path.csv"
        }
        
        # Mock insert_includes
        mock_modified_prompt = "modified prompt content"
        mock_csv_output = "csv,output\nfile1,summary1"
        mock_total_cost = 0.05
        mock_model_name = "test-model"
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path, \
             patch('pdd.auto_deps_main.rprint') as mock_rprint:
            
            mock_construct.return_value = (
                mock_resolved_config,
                mock_input_strings,
                mock_output_file_paths,
                "python"
            )
            
            mock_insert.return_value = (
                mock_modified_prompt,
                mock_csv_output,
                mock_total_cost,
                mock_model_name
            )
            
            # Mock Path operations
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = False
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if arg == "csv_path.csv":
                    return mock_csv_path
                elif arg == "output_path.prompt":
                    return mock_output_path
                return MagicMock()
            
            mock_path.side_effect = path_side_effect
            
            # Execute
            result = auto_deps_main(
                ctx, prompt_file, directory_path, 
                auto_deps_csv_path, output, force_scan
            )
            
            # Verify
            assert result == (mock_modified_prompt, mock_total_cost, mock_model_name)
            
            # Verify construct_paths called correctly
            mock_construct.assert_called_once_with(
                input_file_paths={"prompt_file": prompt_file},
                force=False,
                quiet=False,
                command="auto-deps",
                command_options={"output": output, "csv": auto_deps_csv_path},
                context_override=None,
                confirm_callback=None
            )
            
            # Verify insert_includes called correctly
            mock_insert.assert_called_once_with(
                input_prompt="test prompt content",
                directory_path=directory_path,
                csv_filename="csv_path.csv",
                prompt_filename=prompt_file,
                strength=0.7,
                temperature=0.5,
                time=0.3,
                verbose=True,
                progress_callback=None
            )
            
            # Verify files written
            mock_output_path.write_text.assert_called_once_with(
                mock_modified_prompt, encoding="utf-8"
            )
            mock_csv_path.write_text.assert_called_once_with(
                mock_csv_output, encoding="utf-8"
            )
            
            # Verify output messages
            assert mock_rprint.call_count == 5

    def test_auto_deps_main_with_csv_output(self) -> None:
        """Test that CSV file is written when csv_output is generated."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"}, 
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "csv data", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = False
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "deps.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            assert result == ("modified", 0.01, "model")
            mock_csv_path.write_text.assert_called_once_with("csv data", encoding="utf-8")

    def test_auto_deps_main_without_csv_output(self) -> None:
        """Test that CSV file is not written when csv_output is empty."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            # Empty CSV output
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = False
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "deps.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            assert result == ("modified", 0.01, "model")
            # CSV write should not be called when output is empty
            mock_csv_path.write_text.assert_not_called()


class TestForceScan:
    """Test force_scan flag behavior."""
    
    def test_auto_deps_main_force_scan_deletes_existing_csv(self) -> None:
        """Test that force_scan deletes existing CSV file."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': False}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path, \
             patch('pdd.auto_deps_main.rprint') as mock_rprint:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "csv", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = True
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "deps.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(
                ctx, "prompt.prompt", "dir/", None, None, force_scan=True
            )
            
            # Verify CSV was deleted
            mock_csv_path.unlink.assert_called_once()
            
            # Verify warning message printed
            warning_calls = [c for c in mock_rprint.call_args_list 
                           if 'Removing existing CSV file' in str(c)]
            assert len(warning_calls) == 1

    def test_auto_deps_main_force_scan_no_csv_exists(self) -> None:
        """Test that force_scan doesn't error when CSV doesn't exist."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': False}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path, \
             patch('pdd.auto_deps_main.rprint'):
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = False  # CSV doesn't exist
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "deps.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(
                ctx, "prompt.prompt", "dir/", None, None, force_scan=True
            )
            
            # Should succeed without error
            assert result[0] == "modified"
            # unlink should not be called
            mock_csv_path.unlink.assert_not_called()

    def test_auto_deps_main_no_force_scan_keeps_csv(self) -> None:
        """Test that CSV is not deleted without force_scan flag."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': False}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path, \
             patch('pdd.auto_deps_main.rprint'):
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "csv", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = True
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "deps.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(
                ctx, "prompt.prompt", "dir/", None, None, force_scan=False
            )
            
            # CSV should not be deleted
            mock_csv_path.unlink.assert_not_called()


class TestQuietMode:
    """Test quiet mode behavior."""
    
    def test_auto_deps_main_quiet_mode(self) -> None:
        """Test that no output is printed in quiet mode."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path, \
             patch('pdd.auto_deps_main.rprint') as mock_rprint:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = False
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "deps.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # No output should be printed
            mock_rprint.assert_not_called()
            
            # Verify verbose=False passed to insert_includes
            mock_insert.assert_called_once()
            assert mock_insert.call_args[1]['verbose'] is False

    def test_auto_deps_main_non_quiet_mode(self) -> None:
        """Test that output messages are printed in normal mode."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': False}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path, \
             patch('pdd.auto_deps_main.rprint') as mock_rprint:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "csv", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = False
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "deps.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Should have success messages
            assert mock_rprint.call_count >= 5
            
            # Verify verbose=True passed to insert_includes
            mock_insert.assert_called_once()
            assert mock_insert.call_args[1]['verbose'] is True


class TestContextParameters:
    """Test context parameter handling."""
    
    def test_auto_deps_main_custom_strength_temperature(self) -> None:
        """Test custom strength and temperature values from context."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {
            'force': False,
            'quiet': True,
            'strength': 0.9,
            'temperature': 0.8,
            'time': 0.5
        }
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Verify custom parameters passed to insert_includes
            mock_insert.assert_called_once()
            assert mock_insert.call_args[1]['strength'] == 0.9
            assert mock_insert.call_args[1]['temperature'] == 0.8
            assert mock_insert.call_args[1]['time'] == 0.5

    def test_auto_deps_main_default_parameters(self) -> None:
        """Test default parameter values when not in context."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path, \
             patch('pdd.auto_deps_main.DEFAULT_STRENGTH', 0.5), \
             patch('pdd.auto_deps_main.DEFAULT_TIME', 0.25):
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Verify default parameters used
            mock_insert.assert_called_once()
            assert mock_insert.call_args[1]['strength'] == 0.5
            assert mock_insert.call_args[1]['temperature'] == 0
            assert mock_insert.call_args[1]['time'] == 0.25

    def test_auto_deps_main_with_time_budget(self) -> None:
        """Test time parameter passed correctly."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True, 'time': 0.75}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Verify time parameter passed
            mock_insert.assert_called_once()
            assert mock_insert.call_args[1]['time'] == 0.75


class TestPathHandling:
    """Test path handling and defaults."""
    
    def test_auto_deps_main_custom_output_path(self) -> None:
        """Test custom output path is used."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        custom_output = "custom/output.prompt"
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "resolved_output.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(
                ctx, "prompt.prompt", "dir/", None, custom_output, False
            )
            
            # Verify custom output in command_options
            mock_construct.assert_called_once()
            assert mock_construct.call_args[1]['command_options']['output'] == custom_output

    def test_auto_deps_main_custom_csv_path(self) -> None:
        """Test custom CSV path is used."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        custom_csv = "custom/deps.csv"
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "resolved_csv.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(
                ctx, "prompt.prompt", "dir/", custom_csv, None, False
            )
            
            # Verify custom CSV in command_options
            mock_construct.assert_called_once()
            assert mock_construct.call_args[1]['command_options']['csv'] == custom_csv

    def test_auto_deps_main_default_csv_path(self) -> None:
        """Test default CSV path fallback."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            # construct_paths doesn't return csv in output_file_paths
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt"}, "python"  # No 'csv' key
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            
            mock_csv_path = MagicMock()
            mock_csv_path.exists.return_value = False
            mock_output_path = MagicMock()
            
            def path_side_effect(arg: str) -> MagicMock:
                if "project_dependencies.csv" in str(arg):
                    return mock_csv_path
                return mock_output_path
            
            mock_path.side_effect = path_side_effect
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Verify default CSV path used
            mock_insert.assert_called_once()
            assert mock_insert.call_args[1]['csv_filename'] == "project_dependencies.csv"


class TestExceptionHandling:
    """Test exception handling behavior."""
    
    def test_auto_deps_main_click_abort_reraises(self) -> None:
        """Test that click.Abort exception is re-raised."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct:
            mock_construct.side_effect = click.Abort()
            
            with pytest.raises(click.Abort):
                auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)

    def test_auto_deps_main_general_exception_returns_error(self) -> None:
        """Test that general exceptions return error tuple."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': False}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.rprint') as mock_rprint:
            
            error_msg = "Test error"
            mock_construct.side_effect = Exception(error_msg)
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Should return error tuple
            assert result[0] == ""
            assert result[1] == 0.0
            assert "Error" in result[2]
            assert error_msg in result[2]
            
            # Error should be printed
            error_calls = [c for c in mock_rprint.call_args_list 
                          if 'Error' in str(c)]
            assert len(error_calls) >= 1

    def test_auto_deps_main_exception_with_quiet_mode(self) -> None:
        """Test that exceptions in quiet mode don't print error."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.rprint') as mock_rprint:
            
            mock_construct.side_effect = Exception("Test error")
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Should return error tuple
            assert result[0] == ""
            assert result[1] == 0.0
            assert "Error" in result[2]
            
            # No error should be printed in quiet mode
            mock_rprint.assert_not_called()


class TestIntegration:
    """Test integration with other components."""
    
    def test_auto_deps_main_construct_paths_called_correctly(self) -> None:
        """Test construct_paths is called with correct arguments."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {
            'force': True,
            'quiet': False,
            'context': 'test_context',
            'confirm_callback': Mock()
        }
        
        prompt_file = "test.prompt"
        directory_path = "test_dir/"
        csv_path = "test.csv"
        output = "output.prompt"
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(ctx, prompt_file, directory_path, csv_path, output, False)
            
            # Verify all arguments passed correctly
            mock_construct.assert_called_once_with(
                input_file_paths={"prompt_file": prompt_file},
                force=True,
                quiet=False,
                command="auto-deps",
                command_options={"output": output, "csv": csv_path},
                context_override='test_context',
                confirm_callback=ctx.obj['confirm_callback']
            )

    def test_auto_deps_main_insert_includes_called_correctly(self) -> None:
        """Test insert_includes is called with correct arguments."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {
            'force': False,
            'quiet': False,
            'strength': 0.8,
            'temperature': 0.3,
            'time': 0.4
        }
        
        prompt_file = "test.prompt"
        directory_path = "examples/**/*.py"
        progress_cb = Mock()
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            prompt_content = "original prompt content"
            mock_construct.return_value = (
                {}, {"prompt_file": prompt_content},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(
                ctx, prompt_file, directory_path, None, None, False, progress_cb
            )
            
            # Verify insert_includes called with correct arguments
            mock_insert.assert_called_once_with(
                input_prompt=prompt_content,
                directory_path=directory_path,
                csv_filename="deps.csv",
                prompt_filename=prompt_file,
                strength=0.8,
                temperature=0.3,
                time=0.4,
                verbose=True,
                progress_callback=progress_cb
            )

    def test_auto_deps_main_progress_callback_passed(self) -> None:
        """Test that progress_callback is passed through."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        progress_callback = Mock()
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(
                ctx, "prompt.prompt", "dir/", None, None, False, progress_callback
            )
            
            # Verify callback passed to insert_includes
            mock_insert.assert_called_once()
            assert mock_insert.call_args[1]['progress_callback'] == progress_callback


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_auto_deps_main_none_optional_parameters(self) -> None:
        """Test handling of None optional parameters."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {'force': False, 'quiet': True}
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            mock_path.return_value = MagicMock()
            
            # All optional parameters as None
            result = auto_deps_main(
                ctx, "prompt.prompt", "dir/", 
                auto_deps_csv_path=None, 
                output=None, 
                force_scan=None,
                progress_callback=None
            )
            
            # Should succeed
            assert result[0] == "modified"
            assert result[1] == 0.01
            assert result[2] == "model"

    def test_auto_deps_main_context_override(self) -> None:
        """Test context override is passed correctly."""
        ctx = Mock(spec=click.Context)
        ctx.obj = {
            'force': False,
            'quiet': True,
            'context': 'custom_context'
        }
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Verify context_override passed
            mock_construct.assert_called_once()
            assert mock_construct.call_args[1]['context_override'] == 'custom_context'

    def test_auto_deps_main_confirm_callback(self) -> None:
        """Test confirm_callback is passed correctly."""
        ctx = Mock(spec=click.Context)
        confirm_cb = Mock()
        ctx.obj = {
            'force': False,
            'quiet': True,
            'confirm_callback': confirm_cb
        }
        
        with patch('pdd.auto_deps_main.construct_paths') as mock_construct, \
             patch('pdd.auto_deps_main.insert_includes') as mock_insert, \
             patch('pdd.auto_deps_main.Path') as mock_path:
            
            mock_construct.return_value = (
                {}, {"prompt_file": "content"},
                {"output": "out.prompt", "csv": "deps.csv"}, "python"
            )
            
            mock_insert.return_value = ("modified", "", 0.01, "model")
            mock_path.return_value = MagicMock()
            
            result = auto_deps_main(ctx, "prompt.prompt", "dir/", None, None, False)
            
            # Verify confirm_callback passed
            mock_construct.assert_called_once()
            assert mock_construct.call_args[1]['confirm_callback'] == confirm_cb