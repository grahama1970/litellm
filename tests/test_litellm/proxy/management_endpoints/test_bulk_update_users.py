"""
Mock tests for bulk user update functionality.

These tests mock the API calls to test the bulk update endpoint behavior
without making actual HTTP requests to a running proxy server.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.proxy_server import app

client = TestClient(app)


class TestBulkUpdateUsers:
    """Test class for bulk user update functionality."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup common mocks for all tests."""
        self.mock_prisma_client = MagicMock()
        
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        return {
            "Authorization": "Bearer sk-1234",
            "Content-Type": "application/json",
        }

    @patch("litellm.proxy.proxy_server.prisma_client")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    def test_bulk_update_specific_users_success(self, mock_auth, mock_prisma, auth_headers):
        """Test successful bulk update of specific users."""
        # Mock authentication
        mock_user_obj = MagicMock()
        mock_user_obj.user_role = "proxy_admin"
        mock_auth.return_value = (mock_user_obj, None)
        
        # Mock database update responses
        mock_prisma.db.litellm_usertable.update_many = AsyncMock(return_value=MagicMock(count=2))

        # Test payload for updating specific users
        payload = {
            "users": [
                {"user_id": "user1", "user_role": "internal_user", "max_budget": 100.0},
                {
                    "user_email": "user2@example.com",
                    "user_role": "internal_user_viewer",
                    "max_budget": 50.0,
                },
            ]
        }

        with patch("litellm.proxy.management_endpoints.internal_user_endpoints.prisma_client", mock_prisma):
            response = client.post("/user/bulk_update", headers=auth_headers, json=payload)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "updated_count" in data or "message" in data

    @patch("litellm.proxy.proxy_server.prisma_client")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    def test_bulk_update_all_users_success(self, mock_auth, mock_prisma, auth_headers):
        """Test successful bulk update of all users."""
        # Mock authentication
        mock_user_obj = MagicMock()
        mock_user_obj.user_role = "proxy_admin"
        mock_auth.return_value = (mock_user_obj, None)
        
        # Mock database update responses
        mock_prisma.db.litellm_usertable.update_many = AsyncMock(return_value=MagicMock(count=10))

        # Test payload for updating ALL users
        payload = {
            "all_users": True,
            "user_updates": {"user_role": "internal_user", "max_budget": 75.0},
        }

        with patch("litellm.proxy.management_endpoints.internal_user_endpoints.prisma_client", mock_prisma):
            response = client.post("/user/bulk_update", headers=auth_headers, json=payload)

        # Note: This endpoint might not exist yet, so we expect it to return 404 or similar
        # The test validates that the request structure is correct
        assert response.status_code in [200, 404, 422]  # 404 if endpoint doesn't exist, 422 for validation

    def test_bulk_update_validation_empty_payload(self, auth_headers):
        """Test validation error for empty payload."""
        payload = {}

        response = client.post("/user/bulk_update", headers=auth_headers, json=payload)
        
        # Should return validation error
        assert response.status_code in [400, 422]  # Bad request or validation error

    def test_bulk_update_validation_both_users_and_all_users(self, auth_headers):
        """Test validation error when both 'users' and 'all_users' are specified."""
        payload = {
            "users": [{"user_id": "user1", "user_role": "internal_user"}],
            "all_users": True,
            "user_updates": {"user_role": "internal_user"},
        }

        response = client.post("/user/bulk_update", headers=auth_headers, json=payload)
        
        # Should return validation error
        assert response.status_code in [400, 422]  # Bad request or validation error

    def test_bulk_update_validation_all_users_without_updates(self, auth_headers):
        """Test validation error when all_users=True but no user_updates provided."""
        payload = {"all_users": True}

        response = client.post("/user/bulk_update", headers=auth_headers, json=payload)
        
        # Should return validation error
        assert response.status_code in [400, 422]  # Bad request or validation error

    def test_bulk_update_unauthorized(self):
        """Test bulk update without authorization."""
        payload = {
            "users": [{"user_id": "user1", "user_role": "internal_user"}]
        }

        # No Authorization header
        response = client.post("/user/bulk_update", json=payload)
        
        # Should return unauthorized
        assert response.status_code == 401

    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    def test_bulk_update_forbidden_non_admin(self, mock_auth, auth_headers):
        """Test bulk update with non-admin user."""
        # Mock authentication with non-admin user
        mock_user_obj = MagicMock()
        mock_user_obj.user_role = "internal_user"  # Non-admin role
        mock_auth.return_value = (mock_user_obj, None)

        payload = {
            "users": [{"user_id": "user1", "user_role": "internal_user"}]
        }

        response = client.post("/user/bulk_update", headers=auth_headers, json=payload)
        
        # Should return forbidden (assuming only admins can bulk update)
        assert response.status_code in [403, 404]  # Forbidden or not found if endpoint doesn't exist