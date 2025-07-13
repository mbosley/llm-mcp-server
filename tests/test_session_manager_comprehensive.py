#!/usr/bin/env python3
"""Comprehensive test suite for SessionManager class"""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from session_manager import SessionManager
from utils.feature_flags import FeatureFlags


class TestSessionManager(unittest.TestCase):
    """Comprehensive tests for SessionManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.session_manager = SessionManager(base_dir=self.temp_dir)
        
        # Mock feature flags for consistent testing
        self.feature_flag_patches = [
            patch.object(FeatureFlags, 'is_session_compression_enabled', return_value=False),
            patch.object(FeatureFlags, 'get_session_cache_size', return_value=10),
            patch.object(FeatureFlags, 'get_compression_threshold', return_value=10240)
        ]
        
        for patcher in self.feature_flag_patches:
            patcher.start()
    
    def tearDown(self):
        """Clean up test environment"""
        for patcher in self.feature_flag_patches:
            patcher.stop()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_session(self):
        """Test basic session creation"""
        session = self.session_manager.create_session(
            model="gpt-4",
            system_prompt="Test system prompt",
            metadata={"test": "data"}
        )
        
        # Verify session structure
        self.assertIn("id", session)
        self.assertIn("version", session)
        self.assertIn("created_at", session)
        self.assertIn("updated_at", session)
        self.assertEqual(session["model"], "gpt-4")
        self.assertEqual(session["models_used"], ["gpt-4"])
        self.assertEqual(session["metadata"]["system_prompt"], "Test system prompt")
        self.assertEqual(session["metadata"]["test"], "data")
        self.assertEqual(len(session["messages"]), 1)  # System message
        self.assertEqual(session["messages"][0]["role"], "system")
    
    def test_create_session_custom_id(self):
        """Test session creation with custom ID"""
        custom_id = "test_session_123"
        session = self.session_manager.create_session(
            model="gpt-4",
            session_id=custom_id
        )
        
        self.assertEqual(session["id"], custom_id)
    
    def test_session_id_collision_detection(self):
        """Test session ID collision detection"""
        # Create a session
        session1 = self.session_manager.create_session(model="gpt-4")
        
        # Mock uuid.uuid4().hex to return the same value
        with patch('session_manager.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = session1["id"].split("_")[-1]  # Same hex
            
            # Create another session - should get different ID
            session2 = self.session_manager.create_session(model="gpt-4")
            
            self.assertNotEqual(session1["id"], session2["id"])
    
    def test_load_session(self):
        """Test session loading"""
        # Create a session
        session = self.session_manager.create_session(model="gpt-4")
        session_id = session["id"]
        
        # Load the session
        loaded_session = self.session_manager.load_session(session_id)
        
        self.assertIsNotNone(loaded_session)
        self.assertEqual(loaded_session["id"], session_id)
        self.assertEqual(loaded_session["model"], "gpt-4")
    
    def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist"""
        loaded_session = self.session_manager.load_session("nonexistent")
        self.assertIsNone(loaded_session)
    
    def test_save_session(self):
        """Test session saving"""
        session = self.session_manager.create_session(model="gpt-4")
        
        # Modify session
        session["metadata"]["modified"] = True
        
        # Save session
        self.session_manager.save_session(session)
        
        # Load and verify
        loaded_session = self.session_manager.load_session(session["id"])
        self.assertTrue(loaded_session["metadata"]["modified"])
    
    def test_add_message(self):
        """Test adding messages to session"""
        session = self.session_manager.create_session(model="gpt-4")
        session_id = session["id"]
        
        # Add user message
        updated_session = self.session_manager.add_message(
            session_id, "user", "Hello, AI!", {"test_meta": "data"}
        )
        
        # Verify message was added
        self.assertEqual(len(updated_session["messages"]), 2)  # System + user
        user_message = updated_session["messages"][1]
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(user_message["content"], "Hello, AI!")
        self.assertEqual(user_message["test_meta"], "data")
        
        # Add assistant message with token tracking
        self.session_manager.add_message(
            session_id, "assistant", "Hello, human!",
            metadata={"tokens": {"input": 10, "output": 15, "total": 25}}
        )
        
        # Verify token aggregation
        final_session = self.session_manager.load_session(session_id)
        self.assertEqual(final_session["metadata"]["total_tokens"], 25)
    
    def test_switch_model(self):
        """Test model switching functionality"""
        session = self.session_manager.create_session(model="gpt-4")
        session_id = session["id"]
        
        # Switch model
        updated_session = self.session_manager.switch_model(
            session_id, "claude-3", reason="Better for this task"
        )
        
        # Verify model switch
        self.assertEqual(updated_session["model"], "claude-3")
        self.assertIn("gpt-4", updated_session["models_used"])
        self.assertIn("claude-3", updated_session["models_used"])
        
        # Verify switch message was added
        switch_messages = [m for m in updated_session["messages"] if m.get("model_switch")]
        self.assertEqual(len(switch_messages), 1)
        self.assertIn("Better for this task", switch_messages[0]["content"])
    
    def test_delete_session(self):
        """Test session deletion"""
        session = self.session_manager.create_session(model="gpt-4")
        session_id = session["id"]
        
        # Verify session exists
        self.assertIsNotNone(self.session_manager.load_session(session_id))
        
        # Delete session
        result = self.session_manager.delete_session(session_id)
        self.assertTrue(result)
        
        # Verify session is gone
        self.assertIsNone(self.session_manager.load_session(session_id))
        
        # Try to delete again
        result = self.session_manager.delete_session(session_id)
        self.assertFalse(result)
    
    def test_list_sessions(self):
        """Test session listing functionality"""
        # Create multiple sessions
        session1 = self.session_manager.create_session(model="gpt-4")
        session2 = self.session_manager.create_session(model="claude-3")
        
        # Add tags to one session
        session1["metadata"]["tags"] = ["test", "important"]
        self.session_manager.save_session(session1)
        
        # List all sessions
        all_sessions = self.session_manager.list_sessions()
        self.assertEqual(len(all_sessions), 2)
        
        # List by model
        gpt_sessions = self.session_manager.list_sessions(model="gpt-4")
        self.assertEqual(len(gpt_sessions), 1)
        self.assertEqual(gpt_sessions[0]["model"], "gpt-4")
        
        # List by tag
        tagged_sessions = self.session_manager.list_sessions(tag="test")
        self.assertEqual(len(tagged_sessions), 1)
        self.assertEqual(tagged_sessions[0]["id"], session1["id"])
    
    def test_cache_functionality(self):
        """Test LRU cache functionality"""
        # Create sessions
        sessions = []
        for i in range(15):  # More than cache size
            session = self.session_manager.create_session(model=f"model-{i}")
            sessions.append(session)
        
        # Access some sessions to put them in cache
        for i in range(5):
            self.session_manager.load_session(sessions[i]["id"])
        
        # Cache should only have most recent
        self.assertLessEqual(len(self.session_manager._session_cache), 10)
    
    def test_thread_safety(self):
        """Test thread safety of session operations"""
        session = self.session_manager.create_session(model="gpt-4")
        session_id = session["id"]
        
        # Function to add messages concurrently
        def add_messages(thread_id):
            for i in range(10):
                self.session_manager.add_message(
                    session_id, "user", f"Message {i} from thread {thread_id}"
                )
        
        # Start multiple threads
        threads = []
        for t in range(5):
            thread = threading.Thread(target=add_messages, args=(t,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all messages were added
        final_session = self.session_manager.load_session(session_id)
        # Should have system message + 50 user messages
        self.assertEqual(len(final_session["messages"]), 51)
    
    def test_session_compression(self):
        """Test session compression functionality"""
        # Enable compression for this test
        with patch.object(FeatureFlags, 'is_session_compression_enabled', return_value=True), \
             patch.object(FeatureFlags, 'get_compression_threshold', return_value=100):
            
            # Create a large session
            session = self.session_manager.create_session(model="gpt-4")
            
            # Add a large message to trigger compression
            large_content = "x" * 200  # Larger than threshold
            self.session_manager.add_message(session["id"], "user", large_content)
            
            # Verify we can still load the session
            loaded_session = self.session_manager.load_session(session["id"])
            self.assertIsNotNone(loaded_session)
            
            # Find the large message
            large_messages = [m for m in loaded_session["messages"] if len(m["content"]) > 100]
            self.assertEqual(len(large_messages), 1)
            self.assertEqual(large_messages[0]["content"], large_content)


class TestSessionManagerErrorHandling(unittest.TestCase):
    """Test error handling in SessionManager"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.session_manager = SessionManager(base_dir=self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_message_to_nonexistent_session(self):
        """Test adding message to non-existent session"""
        with self.assertRaises(ValueError):
            self.session_manager.add_message("nonexistent", "user", "Hello")
    
    def test_switch_model_nonexistent_session(self):
        """Test switching model for non-existent session"""
        with self.assertRaises(ValueError):
            self.session_manager.switch_model("nonexistent", "new-model")
    
    def test_corrupted_session_file_handling(self):
        """Test handling of corrupted session files"""
        # Create a corrupted session file
        corrupted_path = Path(self.temp_dir) / "sessions" / "corrupted.json"
        corrupted_path.parent.mkdir(parents=True, exist_ok=True)
        corrupted_path.write_text("invalid json content")
        
        # Loading should return None gracefully
        result = self.session_manager._load_session_from_file(corrupted_path)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()