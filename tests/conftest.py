"""Test configuration and fixtures for pytest"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from session_manager import SessionManager
from utils.feature_flags import FeatureFlags


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def session_manager(temp_dir):
    """Create a SessionManager instance with temporary directory"""
    return SessionManager(base_dir=temp_dir)


@pytest.fixture
def mock_feature_flags():
    """Mock feature flags for consistent testing"""
    with patch.object(FeatureFlags, 'is_unified_sessions_enabled', return_value=True), \
         patch.object(FeatureFlags, 'is_session_compression_enabled', return_value=False), \
         patch.object(FeatureFlags, 'get_session_cache_size', return_value=10), \
         patch.object(FeatureFlags, 'get_compression_threshold', return_value=10240):
        yield


@pytest.fixture
def sample_session(session_manager):
    """Create a sample session for testing"""
    return session_manager.create_session(
        model="gpt-4o",
        system_prompt="You are a helpful assistant.",
        metadata={"test": True}
    )


@pytest.fixture
def sample_messages():
    """Sample message data for testing"""
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
        {"role": "user", "content": "What's the weather like?"},
    ]