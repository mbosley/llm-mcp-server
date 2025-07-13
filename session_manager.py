"""Unified session management for model-agnostic LLM conversations"""

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import OrderedDict
import fcntl
import tempfile
import shutil
import gzip

from utils.feature_flags import FeatureFlags
from utils.logging_config import setup_logging

# Set up logger for this module
logger = setup_logging(__name__)


class SessionManager:
    """Thread-safe unified session manager for all LLM models"""
    
    def __init__(self, base_dir: str = ".llm_sessions"):
        """Initialize the session manager
        
        Args:
            base_dir: Base directory for session storage
        """
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.views_dir = self.base_dir / "views"
        self.index_file = self.base_dir / ".index.sqlite"
        
        # Thread safety
        self._lock = threading.RLock()
        self._cache_lock = threading.RLock()  # Separate lock for cache operations
        self._file_locks = {}
        
        # LRU cache for frequently accessed sessions
        self._cache_size = FeatureFlags.get_session_cache_size()
        self._session_cache = OrderedDict()
        
        # Initialize directory structure
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Create necessary directory structure"""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        (self.views_dir / "by-model").mkdir(parents=True, exist_ok=True)
        (self.views_dir / "by-date").mkdir(parents=True, exist_ok=True)
        (self.views_dir / "by-tag").mkdir(parents=True, exist_ok=True)
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session"""
        return self.sessions_dir / f"{session_id}.json"
    
    def _acquire_file_lock(self, file_path: Path, timeout: float = 5.0) -> Any:
        """Acquire an exclusive file lock with timeout
        
        Args:
            file_path: Path to the file to lock
            timeout: Maximum time to wait for lock
            
        Returns:
            File descriptor with lock acquired
            
        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        start_time = time.time()
        fd = None
        
        try:
            # Ensure file exists
            file_path.touch(exist_ok=True)
            
            # Open file for reading and writing
            fd = os.open(str(file_path), os.O_RDWR | os.O_CREAT)
            
            # Try to acquire exclusive lock
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except IOError:
                    if time.time() - start_time > timeout:
                        if fd is not None:
                            os.close(fd)
                        raise TimeoutError(f"Could not acquire lock on {file_path} within {timeout}s")
                    time.sleep(0.01)
        except Exception:
            if fd is not None:
                os.close(fd)
            raise
    
    def _release_file_lock(self, fd: Any):
        """Release a file lock"""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception:
            pass
    
    def _load_session_from_file(self, session_path: Path) -> Optional[Dict]:
        """Load session from file with proper locking"""
        if not session_path.exists():
            return None
        
        fd = self._acquire_file_lock(session_path)
        try:
            # Read the entire file
            content = os.read(fd, os.path.getsize(fd))
            if not content:
                return None
            
            # Handle compressed sessions
            if session_path.suffix == '.gz' or (
                FeatureFlags.is_session_compression_enabled() and 
                content.startswith(b'\x1f\x8b')
            ):
                content = gzip.decompress(content)
            
            return json.loads(content.decode('utf-8'))
        finally:
            self._release_file_lock(fd)
    
    def _save_session_to_file(self, session_path: Path, session_data: Dict):
        """Save session to file with atomic write and proper locking"""
        # Create temporary file in same directory (for atomic rename)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(session_path.parent),
            prefix=f".{session_path.stem}_",
            suffix=".tmp"
        )
        
        try:
            # Serialize session data
            content = json.dumps(session_data, indent=2, ensure_ascii=False)
            
            # Compress if enabled and session is large
            compression_threshold = FeatureFlags.get_compression_threshold()
            if (FeatureFlags.is_session_compression_enabled() and 
                len(content) > compression_threshold):
                content = gzip.compress(content.encode('utf-8'))
                os.write(temp_fd, content)
            else:
                os.write(temp_fd, content.encode('utf-8'))
            
            os.close(temp_fd)
            
            # Atomic rename
            shutil.move(temp_path, str(session_path))
            
        except Exception:
            try:
                os.close(temp_fd)
                os.unlink(temp_path)
            except Exception:
                pass
            raise
    
    def create_session(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """Create a new unified session
        
        Args:
            model: The initial model for this session
            system_prompt: Optional system prompt
            metadata: Additional metadata
            session_id: Optional custom session ID
            
        Returns:
            Dict containing session data
        """
        with self._lock:
            # Generate session ID if not provided
            if not session_id:
                # Generate unique session ID with collision detection
                max_attempts = 10
                for attempt in range(max_attempts):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_id = f"unified_{timestamp}_{uuid.uuid4().hex[:8]}"
                    
                    # Check if session already exists
                    session_path = self._get_session_path(session_id)
                    if not session_path.exists():
                        break
                else:
                    # If we couldn't find a unique ID after max attempts, use full UUID
                    session_id = f"unified_{timestamp}_{uuid.uuid4().hex}"
            
            # Create unified session structure
            session = {
                "id": session_id,
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "model": model,
                "models_used": [model],
                "messages": [],
                "metadata": {
                    "system_prompt": system_prompt,
                    "tags": [],
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "model_switches": [],
                    **(metadata or {})
                }
            }
            
            # Add system message if provided
            if system_prompt:
                session["messages"].append({
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Save session
            session_path = self._get_session_path(session_id)
            self._save_session_to_file(session_path, session)
            
            # Update cache
            self._update_cache(session_id, session)
            
            # Create view symlinks
            self._create_session_views(session)
            
            logger.info(f"Created new session: {session_id} with model: {model}")
            
            return session
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load an existing session
        
        Args:
            session_id: The session ID to load
            
        Returns:
            Session data or None if not found
        """
        # Check cache first with cache lock
        with self._cache_lock:
            if session_id in self._session_cache:
                # Move to end (LRU) atomically
                self._session_cache.move_to_end(session_id)
                return self._session_cache[session_id].copy()
        
        # Load from file with main lock
        with self._lock:
            # Double-check cache after acquiring main lock
            with self._cache_lock:
                if session_id in self._session_cache:
                    self._session_cache.move_to_end(session_id)
                    return self._session_cache[session_id].copy()
            
            # Load from file
            session_path = self._get_session_path(session_id)
            session = self._load_session_from_file(session_path)
            
            if session:
                self._update_cache(session_id, session)
                logger.debug(f"Loaded session: {session_id}")
            else:
                logger.warning(f"Session not found: {session_id}")
            
            return session
    
    def save_session(self, session: Dict):
        """Save a session to disk
        
        Args:
            session: The session data to save
        """
        with self._lock:
            session_id = session["id"]
            session["updated_at"] = datetime.now().isoformat()
            
            # Save to file
            session_path = self._get_session_path(session_id)
            self._save_session_to_file(session_path, session)
            
            # Update cache
            self._update_cache(session_id, session)
            
            # Update views
            self._update_session_views(session)
            
            logger.debug(f"Saved session: {session_id}")
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Add a message to a session
        
        Args:
            session_id: The session to update
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Updated session data
        """
        with self._lock:
            session = self.load_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Create message
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Add to session
            session["messages"].append(message)
            
            # Update token counts if provided in metadata
            if metadata and "tokens" in metadata:
                if isinstance(metadata["tokens"], dict):
                    # If tokens is a dict with input/output
                    total = metadata["tokens"].get("total", 0)
                    if not total:
                        total = metadata["tokens"].get("input", 0) + metadata["tokens"].get("output", 0)
                    session["metadata"]["total_tokens"] += total
                else:
                    # If tokens is just a number
                    session["metadata"]["total_tokens"] += metadata["tokens"]
            
            # Save session
            self.save_session(session)
            
            logger.debug(f"Added message to session {session_id}: role={role}, length={len(content)}")
            
            return session
    
    def switch_model(
        self,
        session_id: str,
        new_model: str,
        reason: Optional[str] = None
    ) -> Dict:
        """Switch models within a session (B+ approach with tracking)
        
        Args:
            session_id: The session to update
            new_model: The new model to use
            reason: Optional reason for switching
            
        Returns:
            Updated session data
        """
        with self._lock:
            session = self.load_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            old_model = session["model"]
            
            # Track model switch
            switch_event = {
                "timestamp": datetime.now().isoformat(),
                "from_model": old_model,
                "to_model": new_model,
                "reason": reason,
                "message_count": len(session["messages"])
            }
            
            session["model"] = new_model
            
            # Update models used list
            if new_model not in session["models_used"]:
                session["models_used"].append(new_model)
            
            # Add to model switches
            if "model_switches" not in session["metadata"]:
                session["metadata"]["model_switches"] = []
            session["metadata"]["model_switches"].append(switch_event)
            
            # Add system message about switch
            self.add_message(
                session_id,
                "system",
                f"Model switched from {old_model} to {new_model}. Reason: {reason or 'User requested'}",
                metadata={"model_switch": True}
            )
            
            logger.info(f"Switched model for session {session_id}: {old_model} -> {new_model}")
            
            return session
    
    def list_sessions(
        self,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List sessions with optional filtering
        
        Args:
            model: Filter by model
            tag: Filter by tag
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of session summaries
        """
        with self._lock:
            sessions = []
            
            # Get all session files
            session_files = sorted(
                self.sessions_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            for session_file in session_files[offset:offset + limit]:
                try:
                    session = self._load_session_from_file(session_file)
                    if not session:
                        continue
                    
                    # Apply filters
                    if model and model not in session.get("models_used", []):
                        continue
                    
                    if tag and tag not in session.get("metadata", {}).get("tags", []):
                        continue
                    
                    # Create summary
                    summary = {
                        "id": session["id"],
                        "created_at": session["created_at"],
                        "updated_at": session["updated_at"],
                        "model": session["model"],
                        "models_used": session["models_used"],
                        "message_count": len(session["messages"]),
                        "tags": session.get("metadata", {}).get("tags", [])
                    }
                    
                    sessions.append(summary)
                    
                except Exception:
                    # Skip corrupted sessions
                    continue
            
            return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session
        
        Args:
            session_id: The session to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            session_path = self._get_session_path(session_id)
            
            if not session_path.exists():
                return False
            
            # Remove from cache with cache lock
            with self._cache_lock:
                if session_id in self._session_cache:
                    del self._session_cache[session_id]
            
            # Remove file
            session_path.unlink()
            
            # Remove views
            self._remove_session_views(session_id)
            
            logger.info(f"Deleted session: {session_id}")
            
            return True
    
    def _update_cache(self, session_id: str, session: Dict):
        """Update the LRU cache with thread safety"""
        with self._cache_lock:
            # Check if cache size has changed
            current_cache_size = FeatureFlags.get_session_cache_size()
            if current_cache_size != self._cache_size:
                self._cache_size = current_cache_size
            
            # Add to cache
            self._session_cache[session_id] = session.copy()
            self._session_cache.move_to_end(session_id)
            
            # Evict oldest if over capacity
            while len(self._session_cache) > self._cache_size:
                self._session_cache.popitem(last=False)
    
    def _create_session_views(self, session: Dict):
        """Create symlink views for a session"""
        session_id = session["id"]
        session_path = self._get_session_path(session_id)
        
        # By model view
        model = session["model"].replace("/", "_")
        model_dir = self.views_dir / "by-model" / model
        model_dir.mkdir(exist_ok=True)
        model_link = model_dir / f"{session_id}.json"
        try:
            if model_link.exists() or model_link.is_symlink():
                model_link.unlink()
            model_link.symlink_to(f"../../sessions/{session_id}.json")
        except Exception:
            # Symlinks might not work on all systems
            pass
        
        # By date view
        date = datetime.fromisoformat(session["created_at"]).strftime("%Y-%m-%d")
        date_dir = self.views_dir / "by-date" / date
        date_dir.mkdir(exist_ok=True)
        date_link = date_dir / f"{session_id}.json"
        try:
            if date_link.exists() or date_link.is_symlink():
                date_link.unlink()
            date_link.symlink_to(f"../../sessions/{session_id}.json")
        except Exception:
            # Symlinks might not work on all systems
            pass
    
    def _update_session_views(self, session: Dict):
        """Update symlink views when session changes"""
        # For now, just ensure views exist
        self._create_session_views(session)
    
    def _remove_session_views(self, session_id: str):
        """Remove all symlink views for a session"""
        # Remove from all view directories
        for view_type in ["by-model", "by-date", "by-tag"]:
            view_dir = self.views_dir / view_type
            for subdir in view_dir.iterdir():
                if subdir.is_dir():
                    link = subdir / f"{session_id}.json"
                    if link.exists() and link.is_symlink():
                        link.unlink()
    
    def migrate_kimi_session(self, kimi_session_path: Path) -> Dict:
        """Migrate a Kimi session to unified format
        
        Args:
            kimi_session_path: Path to the Kimi session file
            
        Returns:
            Migrated session data
        """
        with open(kimi_session_path, 'r') as f:
            kimi_data = json.load(f)
        
        # Extract session ID from filename
        original_id = kimi_session_path.stem
        
        # Create unified session
        session = self.create_session(
            model="kimi-k2-0711-preview",  # Default Kimi model
            session_id=f"migrated_{original_id}",
            metadata={
                "migrated_from": "kimi",
                "original_id": original_id,
                "migration_date": datetime.now().isoformat()
            }
        )
        
        # Migrate messages
        for msg in kimi_data.get("messages", []):
            self.add_message(
                session["id"],
                msg["role"],
                msg["content"],
                metadata={
                    "original_timestamp": msg.get("timestamp"),
                    "migrated": True
                }
            )
        
        # Migrate metadata
        if "metadata" in kimi_data:
            session["metadata"].update(kimi_data["metadata"])
        
        self.save_session(session)
        return session