import pytest
from unittest.mock import patch, MagicMock
from pdd.context_generator import context_generator
from rich import print

def test_context_generator_unfinished_prompt_exception():
    """
    Test that the context_generator handles exceptions from unfinished_prompt gracefully.
    This test ensures that if unfinished_prompt raises an exception, the generation
    is treated as complete and processing continues.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'test output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.side_effect = Exception("Unfinished prompt error")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, verbose=True
        )
        
        assert example_code == "extracted code"
        assert total_cost == 0.015  # llm cost + postprocess cost
        assert model_name == "test-model"


def test_context_generator_incomplete_generation_with_continue():
    """
    Test that when unfinished_prompt indicates incomplete generation,
    the continue_generation function is called to complete the output.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.continue_generation') as mock_continue, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'incomplete output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.return_value = ("reasoning", False, 0.002, "check-model")
        mock_continue.return_value = ("completed output", 0.015, "continue-model")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, verbose=True
        )
        
        assert example_code == "extracted code"
        assert total_cost == 0.032  # llm + unfinished + continue + postprocess
        assert model_name == "continue-model"
        mock_continue.assert_called_once()


def test_context_generator_complete_generation_no_continue():
    """
    Test that when generation is complete, continue_generation is not called.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.continue_generation') as mock_continue, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'complete output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.return_value = ("reasoning", True, 0.002, "check-model")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, verbose=True
        )
        
        assert example_code == "extracted code"
        assert total_cost == 0.017  # llm + unfinished + postprocess
        assert model_name == "test-model"
        mock_continue.assert_not_called()


def test_context_generator_load_template_failure():
    """
    Test that the function handles failures when loading the prompt template.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load:
        mock_load.return_value = None
        
        code_module = "test_module"
        prompt = "test prompt"
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, verbose=True
        )
        
        assert example_code is None
        assert total_cost == 0.0
        assert model_name is None


def test_context_generator_llm_invoke_exception():
    """
    Test that the function handles exceptions raised by llm_invoke.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.side_effect = Exception("LLM invocation failed")
        
        code_module = "test_module"
        prompt = "test prompt"
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, verbose=True
        )
        
        assert example_code is None
        assert total_cost == 0.0
        assert model_name is None


def test_context_generator_with_optional_parameters():
    """
    Test that optional parameters (source_file_path, example_file_path, module_name)
    are properly passed to llm_invoke.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'test output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.return_value = ("reasoning", True, 0.002, "check-model")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        example_code, total_cost, model_name = context_generator(
            code_module, 
            prompt,
            source_file_path="/path/to/source.py",
            example_file_path="/path/to/example.py",
            module_name="my_module",
            verbose=True
        )
        
        # Verify llm_invoke was called with the optional parameters
        call_args = mock_llm.call_args
        input_json = call_args[1]['input_json']
        assert input_json['source_file_path'] == "/path/to/source.py"
        assert input_json['example_file_path'] == "/path/to/example.py"
        assert input_json['module_name'] == "my_module"
        
        assert example_code == "extracted code"
        assert total_cost > 0
        assert model_name == "test-model"


def test_context_generator_with_custom_time_parameter():
    """
    Test that the custom time parameter is properly passed through to internal functions.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'test output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.return_value = ("reasoning", True, 0.002, "check-model")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        custom_time = 0.8
        
        example_code, total_cost, model_name = context_generator(
            code_module, 
            prompt,
            time=custom_time,
            verbose=False
        )
        
        # Verify time parameter was passed to llm_invoke
        assert mock_llm.call_args[1]['time'] == custom_time
        # Verify time parameter was passed to unfinished_prompt
        assert mock_unfinished.call_args[1]['time'] == custom_time
        # Verify time parameter was passed to postprocess
        assert mock_postprocess.call_args[1]['time'] == custom_time


def test_context_generator_negative_strength():
    """
    Test that negative strength values are rejected.
    """
    code_module = "test_module"
    prompt = "test prompt"
    strength = -0.5
    
    example_code, total_cost, model_name = context_generator(
        code_module, prompt, strength=strength, verbose=True
    )
    
    assert example_code is None
    assert total_cost == 0.0
    assert model_name is None


def test_context_generator_strength_above_one():
    """
    Test that strength values above 1 are rejected.
    """
    code_module = "test_module"
    prompt = "test prompt"
    strength = 1.5
    
    example_code, total_cost, model_name = context_generator(
        code_module, prompt, strength=strength, verbose=True
    )
    
    assert example_code is None
    assert total_cost == 0.0
    assert model_name is None


def test_context_generator_negative_temperature():
    """
    Test that negative temperature values are rejected.
    """
    code_module = "test_module"
    prompt = "test prompt"
    temperature = -0.5
    
    example_code, total_cost, model_name = context_generator(
        code_module, prompt, temperature=temperature, verbose=True
    )
    
    assert example_code is None
    assert total_cost == 0.0
    assert model_name is None


def test_context_generator_with_different_language():
    """
    Test that the language parameter is properly passed through.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'test output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.return_value = ("reasoning", True, 0.002, "check-model")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        language = "javascript"
        
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, language=language, verbose=False
        )
        
        # Verify language was passed to llm_invoke
        assert mock_llm.call_args[1]['input_json']['language'] == language
        # Verify language was passed to unfinished_prompt
        assert mock_unfinished.call_args[1]['language'] == language
        # Verify language was passed to postprocess
        assert mock_postprocess.call_args[1]['language'] == language


def test_context_generator_boundary_strength_values():
    """
    Test that boundary strength values (0 and 1) are accepted.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'test output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.return_value = ("reasoning", True, 0.002, "check-model")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        
        # Test strength = 0
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, strength=0.0, verbose=False
        )
        assert example_code is not None
        
        # Test strength = 1
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, strength=1.0, verbose=False
        )
        assert example_code is not None


def test_context_generator_boundary_temperature_values():
    """
    Test that boundary temperature values (0 and 1) are accepted.
    """
    with patch('pdd.context_generator.load_prompt_template') as mock_load, \
         patch('pdd.context_generator.preprocess') as mock_preprocess, \
         patch('pdd.context_generator.llm_invoke') as mock_llm, \
         patch('pdd.context_generator.unfinished_prompt') as mock_unfinished, \
         patch('pdd.context_generator.postprocess') as mock_postprocess:
        
        mock_load.return_value = "template"
        mock_preprocess.return_value = "processed"
        mock_llm.return_value = {
            'result': 'test output',
            'cost': 0.01,
            'model_name': 'test-model'
        }
        mock_unfinished.return_value = ("reasoning", True, 0.002, "check-model")
        mock_postprocess.return_value = ("extracted code", 0.005, "test-model")
        
        code_module = "test_module"
        prompt = "test prompt"
        
        # Test temperature = 0
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, temperature=0.0, verbose=False
        )
        assert example_code is not None
        
        # Test temperature = 1
        example_code, total_cost, model_name = context_generator(
            code_module, prompt, temperature=1.0, verbose=False
        )
        assert example_code is not None


if __name__ == "__main__":
    pytest.main()