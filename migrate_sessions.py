#!/usr/bin/env python3
"""Migration script for converting Kimi sessions to unified format"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse

from session_manager import SessionManager
from utils.feature_flags import FeatureFlags
from utils.logging_config import setup_logging

# Set up logger for this module
logger = setup_logging(__name__)


class SessionMigrator:
    """Handles migration of legacy sessions to unified format"""
    
    def __init__(self, dry_run: bool = False, backup: bool = True):
        """Initialize the migrator
        
        Args:
            dry_run: If True, only simulate migration without making changes
            backup: If True, create backups before migration
        """
        self.dry_run = dry_run
        self.backup = backup
        self.manager = SessionManager()
        self.stats = {
            "total": 0,
            "migrated": 0,
            "skipped": 0,
            "errors": 0
        }
    
    def migrate_kimi_sessions(self, source_dir: str = ".kimi_sessions") -> Dict:
        """Migrate all Kimi sessions to unified format
        
        Args:
            source_dir: Directory containing Kimi sessions
            
        Returns:
            Migration statistics
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return self.stats
        
        # Get batch size from feature flags
        batch_size = FeatureFlags.get_migration_batch_size()
        
        # Find all Kimi session files
        session_files = list(source_path.glob("*.json"))
        self.stats["total"] = len(session_files)
        
        logger.info(f"Found {len(session_files)} Kimi sessions to migrate")
        logger.info(f"Batch size: {batch_size}, Dry run: {'Yes' if self.dry_run else 'No'}, Backup: {'Yes' if self.backup else 'No'}")
        
        if self.backup and not self.dry_run:
            backup_dir = Path(f"{source_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print(f"\n📦 Creating backup at: {backup_dir}")
            shutil.copytree(source_dir, backup_dir)
        
        print("\n🔄 Starting migration...")
        
        # Process in batches
        for i in range(0, len(session_files), batch_size):
            batch = session_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(session_files) + batch_size - 1) // batch_size
            
            print(f"\n📦 Processing batch {batch_num}/{total_batches}")
            
            for session_file in batch:
                self._migrate_single_session(session_file)
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Migration Summary:")
        print(f"   Total sessions: {self.stats['total']}")
        print(f"   ✅ Migrated: {self.stats['migrated']}")
        print(f"   ⏭️  Skipped: {self.stats['skipped']}")
        print(f"   ❌ Errors: {self.stats['errors']}")
        
        return self.stats
    
    def _migrate_single_session(self, session_file: Path) -> Optional[Dict]:
        """Migrate a single Kimi session file
        
        Args:
            session_file: Path to the session file
            
        Returns:
            Migrated session data or None if failed
        """
        try:
            print(f"   📄 {session_file.name}...", end="")
            
            # Check if already migrated
            migrated_id = f"migrated_{session_file.stem}"
            if self.manager.load_session(migrated_id):
                print(" ⏭️  Already migrated")
                self.stats["skipped"] += 1
                return None
            
            if self.dry_run:
                print(" ✅ Would migrate")
                self.stats["migrated"] += 1
                return None
            
            # Load Kimi session
            with open(session_file, 'r') as f:
                kimi_data = json.load(f)
            
            # Extract metadata
            original_id = session_file.stem
            created_at = None
            
            # Try to extract timestamp from filename or first message
            if '_' in original_id:
                try:
                    # Format: name_YYYYMMDD_HHMM
                    parts = original_id.split('_')
                    if len(parts) >= 3:
                        date_str = parts[-2]
                        time_str = parts[-1]
                        created_at = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
                except Exception:
                    pass
            
            # Use first message timestamp if available
            if not created_at and kimi_data.get("messages"):
                first_msg = kimi_data["messages"][0]
                if "timestamp" in first_msg:
                    try:
                        created_at = datetime.fromisoformat(first_msg["timestamp"])
                    except Exception:
                        pass
            
            # Default to now if no timestamp found
            if not created_at:
                created_at = datetime.now()
            
            # Determine model used
            model = "kimi-k2-0711-preview"  # Default
            if "model" in kimi_data:
                model = kimi_data["model"]
            elif "metadata" in kimi_data and "model" in kimi_data["metadata"]:
                model = kimi_data["metadata"]["model"]
            
            # Extract system prompt
            system_prompt = None
            messages = kimi_data.get("messages", [])
            
            # Remove system message from messages if present
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]
            
            # Create unified session
            session = self.manager.create_session(
                model=model,
                system_prompt=system_prompt,
                session_id=migrated_id,
                metadata={
                    "migrated_from": "kimi",
                    "original_id": original_id,
                    "original_created_at": created_at.isoformat(),
                    "migration_date": datetime.now().isoformat()
                }
            )
            
            # Override created_at to preserve original timestamp
            session["created_at"] = created_at.isoformat()
            
            # Migrate messages
            for msg in messages:
                # Extract message metadata
                msg_metadata = {
                    "migrated": True
                }
                
                # Preserve original timestamp if available
                if "timestamp" in msg:
                    msg_metadata["original_timestamp"] = msg["timestamp"]
                
                # Handle tool calls or other metadata
                if "tool_calls" in msg:
                    msg_metadata["tool_calls"] = msg["tool_calls"]
                
                if "name" in msg:
                    msg_metadata["name"] = msg["name"]
                
                # Add tokens if available
                if "tokens" in msg:
                    msg_metadata["tokens"] = msg["tokens"]
                
                # Add message
                self.manager.add_message(
                    session["id"],
                    msg["role"],
                    msg["content"],
                    metadata=msg_metadata
                )
            
            # Migrate session-level metadata
            if "metadata" in kimi_data:
                # Merge Kimi metadata into unified metadata
                session = self.manager.load_session(session["id"])
                session["metadata"].update({
                    f"kimi_{k}": v for k, v in kimi_data["metadata"].items()
                    if k not in ["model", "system_prompt"]  # Already handled
                })
                self.manager.save_session(session)
            
            print(" ✅ Migrated")
            self.stats["migrated"] += 1
            return session
            
        except Exception as e:
            print(f" ❌ Error: {str(e)}")
            self.stats["errors"] += 1
            return None
    
    def verify_migration(self, original_dir: str = ".kimi_sessions") -> bool:
        """Verify that all sessions were migrated correctly
        
        Args:
            original_dir: Directory containing original sessions
            
        Returns:
            True if all sessions were migrated successfully
        """
        print("\n🔍 Verifying migration...")
        
        original_path = Path(original_dir)
        original_files = list(original_path.glob("*.json"))
        
        verified = 0
        issues = []
        
        for original_file in original_files:
            migrated_id = f"migrated_{original_file.stem}"
            migrated_session = self.manager.load_session(migrated_id)
            
            if not migrated_session:
                issues.append(f"Missing: {original_file.name}")
                continue
            
            # Load original for comparison
            with open(original_file, 'r') as f:
                original_data = json.load(f)
            
            # Compare message counts
            original_messages = original_data.get("messages", [])
            # Skip system messages in count
            original_count = len([m for m in original_messages if m.get("role") != "system"])
            
            migrated_messages = [m for m in migrated_session["messages"] if m.get("role") != "system"]
            migrated_count = len(migrated_messages)
            
            if original_count != migrated_count:
                issues.append(f"Message count mismatch in {original_file.name}: {original_count} vs {migrated_count}")
                continue
            
            verified += 1
        
        print(f"\n✅ Verified: {verified}/{len(original_files)}")
        
        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"   - {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more")
        
        return len(issues) == 0


def main():
    """Main migration script"""
    parser = argparse.ArgumentParser(description="Migrate Kimi sessions to unified format")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups")
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    parser.add_argument("--source", default=".kimi_sessions", help="Source directory for Kimi sessions")
    
    args = parser.parse_args()
    
    print("🔄 Starting Kimi session migration to unified format...")
    
    # Create migrator
    migrator = SessionMigrator(
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
    # Run migration
    stats = migrator.migrate_kimi_sessions(source_dir=args.source)
    
    # Verify if requested
    if args.verify and not args.dry_run and stats["migrated"] > 0:
        migrator.verify_migration(original_dir=args.source)


if __name__ == '__main__':
    main()