#!/usr/bin/env python3
"""Setup script for unified session management"""

import os
from pathlib import Path
from session_manager import SessionManager
from utils.feature_flags import FeatureFlags


def setup_unified_sessions():
    """Initialize the unified session directory structure"""
    
    print("Setting up Unified Session Management")
    print("=" * 50)
    
    # Check if unified sessions are enabled
    if not FeatureFlags.is_unified_sessions_enabled():
        print("\nWARNING: Unified sessions are not enabled!")
        print("To enable, set environment variable: LLM_UNIFIED=1")
        print()
    
    # Create session manager (this will create directories)
    manager = SessionManager()
    
    print(f"\n✅ Created directory structure at: {manager.base_dir}")
    print(f"   - Sessions: {manager.sessions_dir}")
    print(f"   - Views: {manager.views_dir}")
    print(f"     - By Model: {manager.views_dir / 'by-model'}")
    print(f"     - By Date: {manager.views_dir / 'by-date'}")
    print(f"     - By Tag: {manager.views_dir / 'by-tag'}")
    
    # Check for existing Kimi sessions to migrate
    kimi_dir = Path(".kimi_sessions")
    if kimi_dir.exists():
        kimi_sessions = list(kimi_dir.glob("*.json"))
        print(f"\n📁 Found {len(kimi_sessions)} Kimi sessions available for migration")
        print("   Run migration with: python migrate_sessions.py")
    
    # Create .gitignore entries
    gitignore_path = Path(".gitignore")
    gitignore_entries = [
        ".llm_sessions/",
        ".llm_sessions_test/",
        "*.tmp",
        "__pycache__/",
        "*.pyc"
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
        
        entries_to_add = []
        for entry in gitignore_entries:
            if entry not in existing_content:
                entries_to_add.append(entry)
        
        if entries_to_add:
            print(f"\n📝 Adding to .gitignore:")
            with open(gitignore_path, 'a') as f:
                if not existing_content.endswith('\n'):
                    f.write('\n')
                f.write('\n# Unified session management\n')
                for entry in entries_to_add:
                    f.write(f"{entry}\n")
                    print(f"   + {entry}")
    else:
        print(f"\n📝 Creating .gitignore:")
        with open(gitignore_path, 'w') as f:
            f.write("# Unified session management\n")
            for entry in gitignore_entries:
                f.write(f"{entry}\n")
                print(f"   + {entry}")
    
    # Show current feature flags
    print("\n🔧 Current Feature Flags:")
    features = FeatureFlags.get_active_features()
    for feature, value in features.items():
        status = "✅" if value else "❌"
        if isinstance(value, bool):
            print(f"   {status} {feature}")
        else:
            print(f"   📊 {feature}: {value}")
    
    print("\n" + "=" * 50)
    print("Setup complete! 🎉")
    print("\nNext steps:")
    print("1. Enable unified sessions: export LLM_UNIFIED=1")
    print("2. Run tests: python test_session_manager.py")
    print("3. Migrate existing sessions: python migrate_sessions.py")
    print("4. Start using unified sessions in server.py")


if __name__ == '__main__':
    setup_unified_sessions()