"""Prompt constructor with file interpolation support"""

import re
import glob
from pathlib import Path
from typing import Optional, Tuple


def construct_prompt(template: str, base_path: Optional[str] = None) -> str:
    """
    Replace file placeholders in a prompt template with actual file contents.
    
    Supports:
    - {file:path/to/file.py} - Single file inclusion
    - {file:path/to/file.py:10-20} - Line range from file
    - {files:*.py} - Multiple files with glob pattern
    
    Args:
        template: Prompt template with {file:...} placeholders
        base_path: Base directory for relative file paths
        
    Returns:
        Prompt with file contents inserted
    """
    if base_path:
        base_path = Path(base_path)
    else:
        base_path = Path.cwd()
    
    def replace_file_placeholder(match):
        """Replace a single file placeholder with content"""
        placeholder = match.group(1)
        
        # Check if it's a glob pattern (files:)
        if placeholder.startswith('files:'):
            pattern = placeholder[6:]  # Remove 'files:' prefix
            return handle_glob_pattern(pattern, base_path)
        
        # Check if it's a single file (file:)
        if placeholder.startswith('file:'):
            file_spec = placeholder[5:]  # Remove 'file:' prefix
            return handle_single_file(file_spec, base_path)
        
        # Unknown pattern, return as-is
        return match.group(0)
    
    # Pattern to match {file:...} or {files:...}
    pattern = r'\{(files?:[^}]+)\}'
    return re.sub(pattern, replace_file_placeholder, template)


def handle_single_file(file_spec: str, base_path: Path) -> str:
    """Handle single file inclusion with optional line range"""
    # Check for line range
    if ':' in file_spec:
        filepath, line_range = file_spec.rsplit(':', 1)
        if '-' in line_range:
            try:
                start_line, end_line = map(int, line_range.split('-'))
                return read_file_lines(filepath, base_path, start_line, end_line)
            except ValueError:
                # Not a valid line range, treat as part of filename
                filepath = file_spec
    else:
        filepath = file_spec
    
    return read_entire_file(filepath, base_path)


def handle_glob_pattern(pattern: str, base_path: Path) -> str:
    """Handle glob pattern for multiple files"""
    results = []
    
    # Resolve the pattern relative to base path
    full_pattern = str(base_path / pattern)
    matched_files = glob.glob(full_pattern, recursive=True)
    
    if not matched_files:
        return f"\n=== No files matched pattern: {pattern} ===\n"
    
    for filepath in sorted(matched_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get relative path for display
            try:
                display_path = Path(filepath).relative_to(base_path)
            except ValueError:
                display_path = Path(filepath)
            
            results.append(f"\n=== {display_path} ===\n{content}")
        except Exception as e:
            results.append(f"\n=== Error reading {filepath}: {e} ===")
    
    return '\n'.join(results) + '\n'


def read_entire_file(filepath: str, base_path: Path) -> str:
    """Read entire file content"""
    try:
        full_path = base_path / filepath
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"\n=== {filepath} ===\n{content}\n=== end {filepath} ===\n"
    except Exception as e:
        return f"\n=== Error reading {filepath}: {e} ===\n"


def read_file_lines(filepath: str, base_path: Path, start: int, end: int) -> str:
    """Read specific line range from file"""
    try:
        full_path = base_path / filepath
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Adjust to 0-based indexing
        start = max(0, start - 1)
        end = min(len(lines), end)
        
        selected_lines = lines[start:end]
        content = ''.join(selected_lines)
        
        return f"\n=== {filepath} (lines {start+1}-{end}) ===\n{content}\n=== end {filepath} ===\n"
    except Exception as e:
        return f"\n=== Error reading {filepath}: {e} ===\n"


# Convenience function for testing
def preview_prompt(template: str) -> None:
    """Preview how a prompt template will be expanded"""
    print("Template:")
    print(template)
    print("\nExpanded:")
    print(construct_prompt(template))