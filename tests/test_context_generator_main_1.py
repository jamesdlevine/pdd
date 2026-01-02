import sys
import os
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import click
import ast
import httpx

# Adjust path to import the module under test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdd.context_generator_main import context_generator_main, _validate_and_fix_python_syntax

# --- Fixtures ---

@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=click.Context)
    ctx.obj = {}
    return ctx

@pytest.fixture
def mock_construct_paths():
    with patch('pdd.context_generator_main.construct_paths') as mock:
        yield mock

@pytest.fixture
def mock_context_generator():
    with patch('pdd.context_generator_main.context_generator') as mock:
        yield mock

@pytest.fixture
def mock_get_jwt_token():
    with patch('pdd.context_generator_main.CloudConfig.get_jwt_token') as mock_jwt:
        with patch('pdd.context_generator_main.CloudConfig.get_endpoint_url') as mock_url:
            mock_url.return_value = "http://test-cloud-endpoint"
            yield mock_jwt

@pytest.fixture
def mock_httpx_client():
    with patch('httpx.AsyncClient') as mock:
        yield mock

@pytest.fixture
def mock_preprocess():
    with patch('pdd.context_generator_main.preprocess') as mock:
        mock.return_value = "Preprocessed Prompt"
        yield mock

# --- Tests ---

def test_syntax_validation_quiet_false_success(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, tmp_path):
    """Test valid Python syntax with quiet=False (no warnings printed)"""
    mock_ctx.obj['local'] = True
    mock_ctx.obj['quiet'] = False
    
    prompt_file = tmp_path / "test.prompt"
    code_file = tmp_path / "test.py"
    output_file = tmp_path / "test_example.py"
    prompt_file.write_text("Prompt")
    code_file.write_text("Code")
    
    mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
    valid_code = "def hello():\n    return 'world'\n"
    mock_context_generator.return_value = (valid_code, 0.01, "test-model")
    
    result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
    
    assert result_code == valid_code
    assert output_file.read_text() == valid_code


def test_cloud_verbose_mode_short_prompt(mock_ctx, mock_construct_paths, mock_get_jwt_token, mock_httpx_client, mock_preprocess, tmp_path):
    """Test verbose cloud execution with short prompt (no truncation)"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        mock_ctx.obj['verbose'] = True
        mock_ctx.obj['quiet'] = False
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Short prompt")
        code_file.write_text("def foo(): pass")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Short prompt", "code_file": "def foo(): pass"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"generatedExample": "# Cloud", "totalCost": 0.05, "modelName": "gpt-4"}
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
        assert result_code == "# Cloud"


def test_cloud_preprocessing_exception_fallback(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, tmp_path):
    """Test preprocessing exception in cloud triggers local fallback"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        mock_ctx.obj['quiet'] = True
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Prompt")
        code_file.write_text("Code")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        with patch('pdd.context_generator_main.preprocess', side_effect=Exception("Preprocess error")):
            mock_context_generator.return_value = ("# Local", 0.02, "local-model")
            result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
            assert result_code == "# Local"
            mock_context_generator.assert_called_once()


def test_cloud_timeout_not_quiet(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, mock_httpx_client, mock_preprocess, tmp_path):
    """Test cloud timeout with not quiet mode prints warning"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        mock_ctx.obj['quiet'] = False
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Prompt")
        code_file.write_text("Code")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = httpx.TimeoutException("Timeout")
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_context_generator.return_value = ("# Local", 0.02, "local-model")
        result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
        assert result_code == "# Local"


def test_cloud_http_402_error(mock_ctx, mock_construct_paths, mock_get_jwt_token, mock_httpx_client, mock_preprocess, mock_context_generator, tmp_path):
    """Test HTTP 402 (insufficient credits) raises UsageError"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Prompt")
        code_file.write_text("Code")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.text = "Insufficient credits"
        error = httpx.HTTPStatusError("402", request=MagicMock(), response=mock_response)
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = error
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(click.UsageError, match="Insufficient credits"):
            context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)


def test_cloud_http_401_error(mock_ctx, mock_construct_paths, mock_get_jwt_token, mock_httpx_client, mock_preprocess, mock_context_generator, tmp_path):
    """Test HTTP 401 (auth failed) raises UsageError"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Prompt")
        code_file.write_text("Code")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Auth failed"
        error = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_response)
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = error
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(click.UsageError, match="authentication failed"):
            context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)


def test_cloud_http_403_error(mock_ctx, mock_construct_paths, mock_get_jwt_token, mock_httpx_client, mock_preprocess, mock_context_generator, tmp_path):
    """Test HTTP 403 (access denied) raises UsageError"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Prompt")
        code_file.write_text("Code")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Access denied"
        error = httpx.HTTPStatusError("403", request=MagicMock(), response=mock_response)
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = error
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(click.UsageError, match="Access denied"):
            context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)


def test_cloud_http_500_fallback_not_quiet(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, mock_httpx_client, mock_preprocess, tmp_path):
    """Test HTTP 500 falls back to local with warning"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        mock_ctx.obj['quiet'] = False
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Prompt")
        code_file.write_text("Code")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        error = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_response)
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = error
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_context_generator.return_value = ("# Local", 0.02, "local-model")
        result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
        assert result_code == "# Local"


def test_cloud_generic_exception_verbose(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, mock_httpx_client, mock_preprocess, tmp_path):
    """Test generic exception in verbose mode"""
    with patch.dict(os.environ, {"NEXT_PUBLIC_FIREBASE_API_KEY": "fake_key"}):
        mock_ctx.obj['local'] = False
        mock_ctx.obj['verbose'] = True
        
        prompt_file = tmp_path / "test.prompt"
        code_file = tmp_path / "test.py"
        output_file = tmp_path / "test_example.py"
        prompt_file.write_text("Prompt")
        code_file.write_text("Code")
        
        mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
        mock_get_jwt_token.return_value = "fake_jwt_token"
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = Exception("Network error")
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_context_generator.return_value = ("# Local", 0.02, "local-model")
        result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
        assert result_code == "# Local"


def test_success_messages_not_quiet(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, tmp_path):
    """Test success messages printed when not quiet"""
    mock_ctx.obj['local'] = True
    mock_ctx.obj['quiet'] = False
    
    prompt_file = tmp_path / "test.prompt"
    code_file = tmp_path / "test.py"
    output_file = tmp_path / "test_example.py"
    prompt_file.write_text("Prompt")
    code_file.write_text("Code")
    
    mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
    mock_context_generator.return_value = ("# Generated", 0.01234, "gpt-4")
    
    result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
    assert result_code == "# Generated"
    assert cost == 0.01234


def test_exception_reraise_not_quiet(mock_ctx, mock_construct_paths, tmp_path):
    """Test exceptions are re-raised with error message"""
    mock_ctx.obj['quiet'] = False
    
    prompt_file = tmp_path / "test.prompt"
    code_file = tmp_path / "test.py"
    prompt_file.write_text("Prompt")
    code_file.write_text("Code")
    
    mock_construct_paths.side_effect = ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)


def test_output_directory_with_trailing_slash(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, tmp_path):
    """Test output directory path with trailing slash"""
    mock_ctx.obj['local'] = True
    
    prompt_file = tmp_path / "test.prompt"
    code_file = tmp_path / "test.py"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    default_output = output_dir / "example.py"
    
    prompt_file.write_text("Prompt")
    code_file.write_text("Code")
    
    mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(default_output)}, "python")
    mock_context_generator.return_value = ("# Code", 0.01, "model")
    
    result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), str(output_dir) + "/")
    assert result_code == "# Code"
    assert default_output.exists()


def test_non_python_language(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, tmp_path):
    """Test non-Python language skips syntax validation"""
    mock_ctx.obj['local'] = True
    
    prompt_file = tmp_path / "test.prompt"
    code_file = tmp_path / "test.js"
    output_file = tmp_path / "test_example.js"
    prompt_file.write_text("Prompt")
    code_file.write_text("Code")
    
    mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "javascript")
    js_code = "function hello() { return 'world'; }"
    mock_context_generator.return_value = (js_code, 0.01, "model")
    
    result_code, cost, model = context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
    assert result_code == js_code


def test_syntax_fix_backwards_scan(mock_ctx, mock_construct_paths, mock_context_generator, mock_get_jwt_token, tmp_path):
    """Test syntax fixer's backwards scan from end"""
    mock_ctx.obj['local'] = True
    mock_ctx.obj['quiet'] = False
    
    prompt_file = tmp_path / "test.prompt"
    code_file = tmp_path / "test.py"
    output_file = tmp_path / "test_example.py"
    prompt_file.write_text("Prompt")
    code_file.write_text("Code")
    
    mock_construct_paths.return_value = ({}, {"prompt_file": "Prompt", "code_file": "Code"}, {"output": str(output_file)}, "python")
    bad_code = "def hello():\n    print('Hello')\n\n# Comment\ninvalid syntax!!!"
    mock_context_generator.return_value = (bad_code, 0.0, "model")
    
    context_generator_main(mock_ctx, str(prompt_file), str(code_file), None)
    saved = output_file.read_text()
    assert len(saved) > 0