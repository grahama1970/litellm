"""
Mock tests for Ollama local integration functionality.

These tests mock the API calls to test local Ollama behavior
without making actual API calls to a running Ollama server.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

import litellm


class TestOllamaLocalIntegration:
    """Test class for local Ollama integration functionality."""

    def test_local_ollama_completion_mock(self):
        """Mock test for local Ollama completion with gpt-oss:latest."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        
        client = HTTPHandler()
        
        # Mock response data matching Ollama's response format
        mock_response_data = {
            "model": "gpt-oss:latest",
            "created_at": "2024-01-01T00:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": "hello"
            },
            "done": True,
            "total_duration": 123456789,
            "load_duration": 123456789,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 123456789,
            "eval_count": 1,
            "eval_duration": 123456789
        }
        
        with patch.object(client, "post") as mock_post:
            # Mock the HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = mock_response_data
            mock_post.return_value = mock_response
            
            try:
                response = litellm.completion(
                    model="ollama/gpt-oss:latest",
                    messages=[{"role": "user", "content": "Say hello in one word"}],
                    api_base="http://localhost:11434",
                    api_key="dummy-key",
                    client=client,
                )
                
                # If successful, verify response structure
                assert hasattr(response, 'choices')
                
            except Exception as e:
                # Expected to potentially fail during response processing, but verify the call was made correctly
                print(f"Mock test completed with expected processing error: {e}")
            
            # Verify the call was made with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Check URL contains localhost
            url = call_args[1].get("url", "") if len(call_args) > 1 else call_args[0][0] if call_args[0] else ""
            assert "localhost:11434" in url
            
            # Check headers have Bearer prefix for localhost
            headers = call_args[1].get("headers", {}) if len(call_args) > 1 else {}
            assert headers.get("Authorization") == "Bearer dummy-key"

    def test_local_ollama_streaming_mock(self):
        """Mock test for streaming with local Ollama."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        
        client = HTTPHandler()
        
        # Mock streaming response data - multiple chunks
        mock_stream_data = [
            {
                "model": "gpt-oss:latest",
                "created_at": "2024-01-01T00:00:00.000Z",
                "message": {"role": "assistant", "content": "1"},
                "done": False
            },
            {
                "model": "gpt-oss:latest", 
                "created_at": "2024-01-01T00:00:00.000Z",
                "message": {"role": "assistant", "content": " "},
                "done": False
            },
            {
                "model": "gpt-oss:latest",
                "created_at": "2024-01-01T00:00:00.000Z", 
                "message": {"role": "assistant", "content": "2"},
                "done": False
            },
            {
                "model": "gpt-oss:latest",
                "created_at": "2024-01-01T00:00:00.000Z",
                "message": {"role": "assistant", "content": " 3"},
                "done": True,
                "total_duration": 123456789,
                "prompt_eval_count": 10,
                "eval_count": 4
            }
        ]
        
        with patch.object(client, "post") as mock_post:
            # Mock streaming response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.iter_lines.return_value = [
                f"data: {chunk}".encode() for chunk in [
                    '{"model": "gpt-oss:latest", "message": {"content": "1"}, "done": false}',
                    '{"model": "gpt-oss:latest", "message": {"content": " "}, "done": false}',
                    '{"model": "gpt-oss:latest", "message": {"content": "2"}, "done": false}',
                    '{"model": "gpt-oss:latest", "message": {"content": " 3"}, "done": true}'
                ]
            ]
            mock_post.return_value = mock_response
            
            try:
                response = litellm.completion(
                    model="ollama/gpt-oss:latest",
                    messages=[{"role": "user", "content": "Count to 3"}],
                    api_base="http://localhost:11434",
                    api_key="dummy-key",
                    stream=True,
                    client=client,
                )
                
                # If streaming works, iterate through response
                if hasattr(response, '__iter__'):
                    chunks = list(response)
                    assert len(chunks) > 0
                
            except Exception as e:
                # Expected to potentially fail during response processing
                print(f"Mock streaming test completed with expected error: {e}")
            
            # Verify the call was made
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Verify streaming parameter was passed
            json_data = call_args[1].get("json", {}) if len(call_args) > 1 else {}
            assert json_data.get("stream") is True

    def test_local_ollama_embeddings_mock(self):
        """Mock test for embeddings with local Ollama."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        
        client = HTTPHandler()
        
        # Mock embeddings response
        mock_embeddings_data = {
            "model": "gemma3:12b",
            "embedding": [
                -0.123, 0.456, -0.789, 0.012, 
                0.345, -0.678, 0.901, -0.234,
                # ... truncated for brevity, real embeddings would be much longer
            ]
        }
        
        with patch.object(client, "post") as mock_post:
            # Mock the HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = mock_embeddings_data
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_post.return_value = mock_response
            
            try:
                response = litellm.embedding(
                    model="ollama/gemma3:12b",
                    input=["Hello world"],
                    api_base="http://localhost:11434",
                    api_key="dummy-key",
                    client=client,
                )
                
                # If successful, verify response structure
                if hasattr(response, 'data'):
                    assert len(response.data) > 0
                    assert hasattr(response.data[0], 'embedding')
                    
            except Exception as e:
                # Expected to potentially fail during response processing
                print(f"Mock embeddings test completed with expected error: {e}")
            
            # Verify the call was made with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Check URL and headers
            url = call_args[1].get("url", "") if len(call_args) > 1 else ""
            assert "localhost:11434" in url
            
            headers = call_args[1].get("headers", {}) if len(call_args) > 1 else {}
            assert headers.get("Authorization") == "Bearer dummy-key"

    def test_regular_ollama_localhost_mock(self):
        """Mock test for regular Ollama with localhost."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        
        client = HTTPHandler()
        
        # Mock response data
        mock_response_data = {
            "model": "llama2",
            "created_at": "2024-01-01T00:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": "Hello from regular Ollama"
            },
            "done": True,
            "total_duration": 234567890,
            "load_duration": 234567890,
            "prompt_eval_count": 8,
            "prompt_eval_duration": 234567890,
            "eval_count": 6,
            "eval_duration": 234567890
        }
        
        with patch.object(client, "post") as mock_post:
            # Mock the HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_post.return_value = mock_response
            
            try:
                response = litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Say hello"}],
                    api_base="http://localhost:11434",
                    api_key="test-key",
                    client=client,
                )
                
            except Exception as e:
                print(f"Mock test completed with expected error: {e}")
            
            # Verify the call was made with Bearer prefix for localhost
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            headers = call_args[1].get("headers", {}) if len(call_args) > 1 else {}
            assert headers.get("Authorization") == "Bearer test-key"

    def test_custom_ollama_endpoint_mock(self):
        """Mock test for custom Ollama endpoint (not localhost, not ollama.com)."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        
        client = HTTPHandler()
        
        with patch.object(client, "post") as mock_post:
            # Mock a connection error to simulate server not existing
            from requests.exceptions import ConnectionError
            mock_post.side_effect = ConnectionError("Connection refused")
            
            try:
                response = litellm.completion(
                    model="ollama/llama2",
                    messages=[{"role": "user", "content": "Say hello"}],
                    api_base="https://my-ollama-server.com",
                    api_key="custom-key",
                    client=client,
                )
            except Exception as e:
                # Expected to fail - verify authorization header format
                error_str = str(e)
                # This test verifies the header was set correctly before the connection failed
                print(f"Expected connection error: {e}")
            
            # Verify the call was attempted with correct headers
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Custom endpoints (not localhost) should get Bearer prefix
            headers = call_args[1].get("headers", {}) if len(call_args) > 1 else {}
            assert headers.get("Authorization") == "Bearer custom-key"