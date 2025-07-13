"""Feature flag management for incremental rollout of model-agnostic sessions"""

import os
from typing import Optional


class FeatureFlags:
    """Manage feature flags for incremental rollout"""
    
    @staticmethod
    def is_unified_sessions_enabled() -> bool:
        """Check if unified session management is enabled
        
        Enable by setting environment variable:
        - LLM_UNIFIED=1
        - LLM_UNIFIED=true
        - LLM_UNIFIED=yes
        
        Returns:
            bool: True if unified sessions are enabled
        """
        return os.getenv('LLM_UNIFIED', '').lower() in ('1', 'true', 'yes')
    
    @staticmethod
    def is_legacy_mode() -> bool:
        """Check if forced legacy mode is enabled
        
        Force legacy mode by setting environment variable:
        - LLM_LEGACY=1
        - LLM_LEGACY=true
        - LLM_LEGACY=yes
        
        This overrides LLM_UNIFIED if both are set.
        
        Returns:
            bool: True if legacy mode is forced
        """
        return os.getenv('LLM_LEGACY', '').lower() in ('1', 'true', 'yes')
    
    @staticmethod
    def is_dual_write_enabled() -> bool:
        """Check if dual-write mode is enabled
        
        In dual-write mode, sessions are written to both old and new formats
        for safety during migration period.
        
        Enable by setting environment variable:
        - LLM_DUAL_WRITE=1
        - LLM_DUAL_WRITE=true
        - LLM_DUAL_WRITE=yes
        
        Returns:
            bool: True if dual-write mode is enabled
        """
        return os.getenv('LLM_DUAL_WRITE', '').lower() in ('1', 'true', 'yes')
    
    @staticmethod
    def get_migration_batch_size() -> int:
        """Get batch size for session migration
        
        Control migration batch size with:
        - LLM_MIGRATION_BATCH_SIZE=100 (default: 50)
        
        Returns:
            int: Number of sessions to migrate in each batch
        """
        try:
            return int(os.getenv('LLM_MIGRATION_BATCH_SIZE', '50'))
        except ValueError:
            return 50
    
    @staticmethod
    def is_session_compression_enabled() -> bool:
        """Check if session compression is enabled
        
        Enable compression for large sessions:
        - LLM_SESSION_COMPRESS=1
        - LLM_SESSION_COMPRESS=true
        - LLM_SESSION_COMPRESS=yes
        
        Returns:
            bool: True if session compression is enabled
        """
        return os.getenv('LLM_SESSION_COMPRESS', '').lower() in ('1', 'true', 'yes')
    
    @staticmethod
    def get_session_cache_size() -> int:
        """Get size of in-memory session cache
        
        Control cache size with:
        - LLM_SESSION_CACHE_SIZE=100 (default: 50)
        
        Returns:
            int: Maximum number of sessions to cache in memory
        """
        try:
            return int(os.getenv('LLM_SESSION_CACHE_SIZE', '50'))
        except ValueError:
            return 50
    
    @staticmethod
    def get_compression_threshold() -> int:
        """Get size threshold for session compression
        
        Control compression threshold with:
        - LLM_COMPRESSION_THRESHOLD=10240 (default: 10240 bytes = 10KB)
        
        Returns:
            int: Size in bytes above which sessions will be compressed
        """
        try:
            return int(os.getenv('LLM_COMPRESSION_THRESHOLD', '10240'))
        except ValueError:
            return 10240
    
    @staticmethod
    def is_sqlite_index_enabled() -> bool:
        """Check if SQLite session indexing is enabled
        
        Enable SQLite indexing for fast session search:
        - LLM_SESSION_INDEX=1
        - LLM_SESSION_INDEX=true
        - LLM_SESSION_INDEX=yes
        
        Returns:
            bool: True if SQLite indexing is enabled
        """
        return os.getenv('LLM_SESSION_INDEX', '').lower() in ('1', 'true', 'yes')
    
    @staticmethod
    def get_active_features() -> dict:
        """Get summary of all active feature flags
        
        Returns:
            dict: Dictionary of feature names and their states
        """
        return {
            'unified_sessions': FeatureFlags.is_unified_sessions_enabled(),
            'legacy_mode': FeatureFlags.is_legacy_mode(),
            'dual_write': FeatureFlags.is_dual_write_enabled(),
            'session_compression': FeatureFlags.is_session_compression_enabled(),
            'compression_threshold': FeatureFlags.get_compression_threshold(),
            'sqlite_index': FeatureFlags.is_sqlite_index_enabled(),
            'cache_size': FeatureFlags.get_session_cache_size(),
            'migration_batch_size': FeatureFlags.get_migration_batch_size(),
        }