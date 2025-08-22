import asyncio
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest import mock

import pytest

import litellm


def test_ollama_turbo_auth_header():
    """Test that ollama.com URLs get the correct auth header without Bearer prefix."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    from unittest.mock import patch

    with patch.object(client, "post") as mock_post:
        try:
            litellm.completion(
                model="ollama_chat/gpt-oss:120b",
                messages=[{"role": "user", "content": "test"}],
                api_base="https://ollama.com",
                api_key="test_key_123",
                client=client,
            )
        except Exception as e:
            print(e)
        
        mock_post.assert_called()
        
        # Check the headers just like test_ollama.py does
        print(mock_post.call_args.kwargs)
        headers = mock_post.call_args.kwargs.get("headers", {})
        
        # Should NOT have Bearer prefix for ollama.com
        assert headers.get("Authorization") == "test_key_123"
        assert "Bearer" not in headers.get("Authorization", "")


def test_ollama_localhost_auth_header():
    """Test that localhost URLs get the correct auth header with Bearer prefix."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    from unittest.mock import patch

    with patch.object(client, "post") as mock_post:
        try:
            litellm.completion(
                model="ollama_chat/llama2",
                messages=[{"role": "user", "content": "test"}],
                api_base="http://localhost:11434",
                api_key="test_key_456",
                client=client,
            )
        except Exception as e:
            print(e)
        
        mock_post.assert_called()
        
        # Check the headers just like test_ollama.py does
        print(mock_post.call_args.kwargs)
        headers = mock_post.call_args.kwargs.get("headers", {})
        
        # Should have Bearer prefix for localhost
        assert headers.get("Authorization") == "Bearer test_key_456"


@pytest.mark.skip(reason="Integration test - requires OLLAMA_TURBO_API_KEY")
def test_ollama_turbo_integration():
    """Integration test with real Ollama Turbo API."""
    api_key = os.environ.get("OLLAMA_TURBO_API_KEY")
    if not api_key:
        pytest.skip("OLLAMA_TURBO_API_KEY not set")
    
    try:
        response = litellm.completion(
            model="ollama_chat/gpt-oss:120b",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            api_base="https://ollama.com",
            api_key=api_key,
            max_tokens=10,
        )
        
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        
    except Exception as e:
        pytest.fail(f"Ollama Turbo API call failed: {str(e)}")


def test_ollama_turbo_mock_completion():
    """Mock test for Ollama Turbo completion functionality."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    
    client = HTTPHandler()
    from unittest.mock import patch
    
    # Mock response data
    mock_response_data = {
        "model": "gpt-oss:120b", 
        "created_at": "2024-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant",
            "content": "Hello"
        },
        "done": True,
        "total_duration": 123456789,
        "load_duration": 123456789,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 123456789,
        "eval_count": 5,
        "eval_duration": 123456789
    }
    
    with patch.object(client, "post") as mock_post:
        # Mock the HTTP response
        mock_post.return_value.json.return_value = mock_response_data
        mock_post.return_value.status_code = 200
        mock_post.return_value.headers = {"content-type": "application/json"}
        
        try:
            response = litellm.completion(
                model="ollama_chat/gpt-oss:120b",
                messages=[{"role": "user", "content": "Say hello"}],
                api_base="https://ollama.com",
                api_key="mock-api-key",
                client=client,
            )
        except Exception as e:
            # We expect this to potentially fail due to response processing
            print(f"Expected exception during mock test: {e}")
        
        # Verify the call was made with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check URL
        assert "ollama.com" in call_args[1]["url"] or "ollama.com" in call_args[0][0]
        
        # Check headers
        headers = call_args[1].get("headers", {}) if call_args[1] else {}
        assert headers.get("Authorization") == "mock-api-key"  # No Bearer prefix for ollama.com


def test_ollama_local_mock_completion():
    """Mock test for local Ollama completion functionality."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    
    client = HTTPHandler()
    from unittest.mock import patch
    
    # Mock response data for local Ollama
    mock_response_data = {
        "model": "llama2", 
        "created_at": "2024-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant",
            "content": "Hello from local Ollama"
        },
        "done": True,
        "total_duration": 123456789,
        "load_duration": 123456789,
        "prompt_eval_count": 15,
        "prompt_eval_duration": 123456789,
        "eval_count": 8,
        "eval_duration": 123456789
    }
    
    with patch.object(client, "post") as mock_post:
        # Mock the HTTP response
        mock_post.return_value.json.return_value = mock_response_data
        mock_post.return_value.status_code = 200
        mock_post.return_value.headers = {"content-type": "application/json"}
        
        try:
            response = litellm.completion(
                model="ollama_chat/llama2",
                messages=[{"role": "user", "content": "Say hello"}],
                api_base="http://localhost:11434",
                api_key="dummy-key",
                client=client,
            )
        except Exception as e:
            # We expect this to potentially fail due to response processing
            print(f"Expected exception during mock test: {e}")
        
        # Verify the call was made with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check URL
        assert "localhost:11434" in call_args[1]["url"] or "localhost:11434" in call_args[0][0]
        
        # Check headers (should have Bearer prefix for localhost)
        headers = call_args[1].get("headers", {}) if call_args[1] else {}
        assert headers.get("Authorization") == "Bearer dummy-key"


def test_ollama_custom_endpoint_mock():
    """Mock test for custom Ollama endpoint functionality."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    
    client = HTTPHandler()
    from unittest.mock import patch
    
    with patch.object(client, "post") as mock_post:
        # Mock a connection error to simulate custom server not existing
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("Connection failed")
        
        try:
            response = litellm.completion(
                model="ollama_chat/llama2",
                messages=[{"role": "user", "content": "Say hello"}],
                api_base="https://my-custom-ollama.com",
                api_key="custom-key",
                client=client,
            )
        except Exception as e:
            # Expected to fail - check that the authorization header was set correctly
            print(f"Expected connection error: {e}")
        
        # Verify the call was attempted with correct headers
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check that Bearer prefix was added for non-localhost custom endpoint
        headers = call_args[1].get("headers", {}) if call_args[1] else {}
        assert headers.get("Authorization") == "Bearer custom-key"


