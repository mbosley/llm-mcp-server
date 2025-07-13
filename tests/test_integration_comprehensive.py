#!/usr/bin/env python3
"""Comprehensive integration tests for multi-model conversation scenarios"""

import os
import tempfile
import unittest
from unittest.mock import patch, Mock
from pathlib import Path

from session_manager import SessionManager
from adapters.openai_adapter import OpenAIAdapter
from adapters.gemini_adapter import GeminiAdapter
from adapters.kimi_adapter import KimiAdapter
from utils.feature_flags import FeatureFlags


class TestMultiModelIntegration(unittest.TestCase):
    """Test multi-model conversation scenarios"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.session_manager = SessionManager(base_dir=self.temp_dir)
        
        # Mock feature flags
        self.feature_flag_patches = [
            patch.object(FeatureFlags, 'is_unified_sessions_enabled', return_value=True),
            patch.object(FeatureFlags, 'is_session_compression_enabled', return_value=False),
            patch.object(FeatureFlags, 'get_session_cache_size', return_value=10)
        ]
        
        for patcher in self.feature_flag_patches:
            patcher.start()
    
    def tearDown(self):
        """Clean up test environment"""
        for patcher in self.feature_flag_patches:
            patcher.stop()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_and_switch_models(self):
        """Test creating session and switching between models"""
        # Start with OpenAI
        session = self.session_manager.create_session(
            model="gpt-4o",
            system_prompt="You are a helpful assistant."
        )
        session_id = session["id"]
        
        # Add a user message
        self.session_manager.add_message(session_id, "user", "Hello!")
        
        # Switch to Gemini
        self.session_manager.switch_model(session_id, "gemini-2.5-pro", "Better for analysis")
        
        # Add another message
        self.session_manager.add_message(session_id, "assistant", "Hello! How can I help?")
        
        # Switch to Kimi
        self.session_manager.switch_model(session_id, "kimi-k2-0711-preview", "Testing Kimi")
        
        # Load final session and verify
        final_session = self.session_manager.load_session(session_id)
        
        # Should have all models tracked
        self.assertEqual(final_session["model"], "kimi-k2-0711-preview")
        self.assertIn("gpt-4o", final_session["models_used"])
        self.assertIn("gemini-2.5-pro", final_session["models_used"])
        self.assertIn("kimi-k2-0711-preview", final_session["models_used"])
        
        # Should have system message + user message + assistant message + 2 switch messages
        self.assertEqual(len(final_session["messages"]), 5)
        
        # Check for switch messages
        switch_messages = [m for m in final_session["messages"] if m.get("model_switch")]
        self.assertEqual(len(switch_messages), 2)
    
    def test_session_persistence_across_restarts(self):
        """Test that sessions persist across SessionManager restarts"""
        # Create a session
        session = self.session_manager.create_session(model="gpt-4o")
        session_id = session["id"]
        
        # Add messages
        self.session_manager.add_message(session_id, "user", "First message")
        self.session_manager.add_message(session_id, "assistant", "First response")
        
        # Create new SessionManager instance (simulating restart)
        new_session_manager = SessionManager(base_dir=self.temp_dir)
        
        # Load session with new instance
        loaded_session = new_session_manager.load_session(session_id)
        
        self.assertIsNotNone(loaded_session)
        self.assertEqual(loaded_session["id"], session_id)
        self.assertEqual(loaded_session["model"], "gpt-4o")
        self.assertEqual(len(loaded_session["messages"]), 3)  # system + user + assistant
    
    def test_concurrent_session_access(self):
        """Test concurrent access to the same session"""
        import threading
        import time
        
        # Create a session
        session = self.session_manager.create_session(model="gpt-4o")
        session_id = session["id"]
        
        results = []
        errors = []
        
        def add_messages(thread_id, count):
            try:
                for i in range(count):
                    self.session_manager.add_message(
                        session_id, "user", f"Message {i} from thread {thread_id}"
                    )
                    time.sleep(0.01)  # Small delay
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Start multiple threads
        threads = []
        for t in range(3):
            thread = threading.Thread(target=add_messages, args=(t, 5))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 3)
        
        # Verify all messages were added
        final_session = self.session_manager.load_session(session_id)
        self.assertEqual(len(final_session["messages"]), 16)  # 1 system + 15 user messages
    
    def test_large_conversation_handling(self):
        """Test handling of large conversations"""
        session = self.session_manager.create_session(model="gpt-4o")
        session_id = session["id"]
        
        # Add many messages
        for i in range(100):
            self.session_manager.add_message(session_id, "user", f"Message {i}")
            if i % 10 == 0:  # Switch models occasionally
                new_model = ["gemini-2.5-flash", "kimi-k2-0711-preview", "gpt-4o-mini"][i // 30 % 3]
                self.session_manager.switch_model(session_id, new_model)
        
        # Load and verify
        final_session = self.session_manager.load_session(session_id)
        
        # Should handle large session without issues
        self.assertGreater(len(final_session["messages"]), 100)
        self.assertGreater(len(final_session["models_used"]), 1)
    
    def test_session_metadata_tracking(self):
        """Test comprehensive metadata tracking across model switches"""
        session = self.session_manager.create_session(
            model="gpt-4o",
            metadata={"project": "test", "user": "alice"}
        )
        session_id = session["id"]
        
        # Add messages with token tracking
        self.session_manager.add_message(
            session_id, "user", "Hello",
            metadata={"tokens": {"input": 5, "output": 0, "total": 5}}
        )
        
        self.session_manager.add_message(
            session_id, "assistant", "Hi there!",
            metadata={"tokens": {"input": 0, "output": 10, "total": 10}}
        )
        
        # Switch model
        self.session_manager.switch_model(session_id, "gemini-2.5-pro")
        
        # Add more messages
        self.session_manager.add_message(
            session_id, "assistant", "How can I help?",
            metadata={"tokens": {"input": 0, "output": 15, "total": 15}}
        )
        
        # Load and verify metadata
        final_session = self.session_manager.load_session(session_id)
        
        # Check custom metadata preserved
        self.assertEqual(final_session["metadata"]["project"], "test")
        self.assertEqual(final_session["metadata"]["user"], "alice")
        
        # Check token aggregation
        self.assertEqual(final_session["metadata"]["total_tokens"], 30)
        
        # Check model switches tracked
        self.assertGreater(len(final_session["metadata"]["model_switches"]), 0)
    
    def test_session_listing_and_filtering(self):
        """Test session listing with various filters"""
        # Create multiple sessions with different models and tags
        session1 = self.session_manager.create_session(
            model="gpt-4o",
            metadata={"tags": ["work", "important"]}
        )
        
        session2 = self.session_manager.create_session(
            model="gemini-2.5-pro",
            metadata={"tags": ["personal"]}
        )
        
        session3 = self.session_manager.create_session(model="kimi-k2-0711-preview")
        
        # Switch model in session1 to add gemini to models_used
        self.session_manager.switch_model(session1["id"], "gemini-2.5-flash")
        
        # Test listing all sessions
        all_sessions = self.session_manager.list_sessions()
        self.assertEqual(len(all_sessions), 3)
        
        # Test filtering by model
        gpt_sessions = self.session_manager.list_sessions(model="gpt-4o")
        self.assertEqual(len(gpt_sessions), 1)
        self.assertEqual(gpt_sessions[0]["id"], session1["id"])
        
        # Test filtering by tag
        work_sessions = self.session_manager.list_sessions(tag="work")
        self.assertEqual(len(work_sessions), 1)
        self.assertEqual(work_sessions[0]["id"], session1["id"])
    
    def test_session_deletion_cleanup(self):
        """Test that session deletion properly cleans up all artifacts"""
        # Create session
        session = self.session_manager.create_session(model="gpt-4o")
        session_id = session["id"]
        
        # Add messages to create session views
        self.session_manager.add_message(session_id, "user", "Test message")
        
        # Verify session exists
        self.assertIsNotNone(self.session_manager.load_session(session_id))
        
        # Delete session
        result = self.session_manager.delete_session(session_id)
        self.assertTrue(result)
        
        # Verify session is gone
        self.assertIsNone(self.session_manager.load_session(session_id))
        
        # Verify session file is gone
        session_path = self.session_manager._get_session_path(session_id)
        self.assertFalse(session_path.exists())


class TestFeatureFlagIntegration(unittest.TestCase):
    """Test integration with feature flags"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_unified_sessions_disabled(self):
        """Test behavior when unified sessions are disabled"""
        with patch.object(FeatureFlags, 'is_unified_sessions_enabled', return_value=False):
            # Should still be able to create SessionManager
            session_manager = SessionManager(base_dir=self.temp_dir)
            
            # But unified features might behave differently
            session = session_manager.create_session(model="gpt-4o")
            self.assertIsNotNone(session)
    
    def test_compression_enabled(self):
        """Test session compression when enabled"""
        with patch.object(FeatureFlags, 'is_session_compression_enabled', return_value=True), \
             patch.object(FeatureFlags, 'get_compression_threshold', return_value=100):
            
            session_manager = SessionManager(base_dir=self.temp_dir)
            session = session_manager.create_session(model="gpt-4o")
            
            # Add large content to trigger compression
            large_content = "x" * 200
            session_manager.add_message(session["id"], "user", large_content)
            
            # Should still be able to load normally
            loaded_session = session_manager.load_session(session["id"])
            self.assertIsNotNone(loaded_session)
            
            # Find the large message
            large_messages = [m for m in loaded_session["messages"] if len(m["content"]) > 100]
            self.assertEqual(len(large_messages), 1)
            self.assertEqual(large_messages[0]["content"], large_content)


if __name__ == "__main__":
    unittest.main()