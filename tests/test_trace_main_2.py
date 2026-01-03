import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import sys

from pdd.trace import trace
from pdd import DEFAULT_STRENGTH, DEFAULT_TIME

@pytest.fixture
def mock_deps():
    with patch('pdd.trace.load_prompt_template') as mock_load, \
         patch('pdd.trace.preprocess') as mock_preprocess, \
         patch('pdd.trace.llm_invoke') as mock_llm_invoke:
        yield mock_load, mock_preprocess, mock_llm_invoke

def test_trace_happy_path(mock_deps):
    mock_load, mock_preprocess, mock_llm_invoke = mock_deps

    # Setup mocks
    mock_load.return_value = "template"
    mock_preprocess.return_value = "processed_template"

    # First LLM call (trace_LLM)
    trace_response = {
        'cost': 0.001,
        'model_name': 'gpt-4',
        'result': 'analysis result'
    }
    
    # Second LLM call (extract_promptline_LLM)
    # We use SimpleNamespace to mock the Pydantic object structure
    # expected by the trace function (obj.prompt_line)
    extract_response = {
        'cost': 0.002,
        'result': SimpleNamespace(prompt_line="This is the target line")
    }

    mock_llm_invoke.side_effect = [trace_response, extract_response]

    code_file = "def test():\n    pass"
    code_line = 1
    prompt_file = "Line 1\nThis is the target line\nLine 3"

    # Execute
    result_line, total_cost, model_name = trace(
        code_file, code_line, prompt_file
    )

    # Verify
    assert result_line == 2
    assert total_cost == 0.003
    assert model_name == 'gpt-4'
    
    # Verify calls
    assert mock_load.call_count == 2
    assert mock_preprocess.call_count == 2
    assert mock_llm_invoke.call_count == 2

def test_trace_invalid_input(mock_deps):
    # Test with invalid line number
    code_file = "line1"
    prompt_file = "prompt"
    
    # Should return fallback (1) and 0 cost
    line, cost, model = trace(code_file, 99, prompt_file)
    assert line == 1
    assert cost == 0.0
    assert model == "fallback"

def test_trace_llm_error(mock_deps):
    mock_load, _, mock_llm_invoke = mock_deps
    mock_load.return_value = "template"
    
    # Simulate LLM error
    mock_llm_invoke.side_effect = Exception("LLM Error")
    
    code_file = "code"
    prompt_file = "prompt"
    
    line, cost, model = trace(code_file, 1, prompt_file)
    
    # Should handle error and return fallback
    assert line == 1
    assert cost == 0.0
    assert model == "fallback"

def test_trace_fuzzy_match_logic(mock_deps):
    mock_load, mock_preprocess, mock_llm_invoke = mock_deps
    mock_load.return_value = "template"
    mock_preprocess.return_value = "processed"
    
    # Setup LLM to return a slightly different string
    trace_response = {'cost': 0, 'model_name': 'gpt-4', 'result': ''}
    # The LLM returns "target line" but the file has "Target Line."
    extract_response = {'cost': 0, 'result': SimpleNamespace(prompt_line="target line")}
    
    mock_llm_invoke.side_effect = [trace_response, extract_response]
    
    code_file = "code"
    prompt_file = "Line 1\nTarget Line.\nLine 3"
    
    line, _, _ = trace(code_file, 1, prompt_file)
    
    # Should match line 2 due to fuzzy matching normalization
    assert line == 2