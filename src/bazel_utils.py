#!/usr/bin/env python3
"""
Bazel Utilities for Path Resolution

Provides utilities to resolve paths correctly when running under Bazel vs direct Python execution.
Bazel sets environment variables that help us find the workspace root.
"""

import os
from pathlib import Path
from typing import Union


def get_workspace_root() -> Path:
    """Get the workspace root directory, handling both Bazel and direct execution.
    
    When running under Bazel, use BUILD_WORKSPACE_DIRECTORY or detect from runfiles.
    When running directly, use the current working directory or find workspace root.
    
    Returns:
        Path object pointing to the workspace root
    """
    # Method 1: Bazel sets BUILD_WORKSPACE_DIRECTORY when using `bazel run`
    if 'BUILD_WORKSPACE_DIRECTORY' in os.environ:
        workspace_dir = os.environ['BUILD_WORKSPACE_DIRECTORY']
        return Path(workspace_dir)
    
    # Method 2: Look for RUNFILES_DIR (when running under Bazel test/run)
    if 'RUNFILES_DIR' in os.environ:
        runfiles_dir = Path(os.environ['RUNFILES_DIR'])
        # The workspace should be under runfiles_dir/eval_annotate_scale
        workspace_candidate = runfiles_dir / "eval_annotate_scale"
        if workspace_candidate.exists():
            return workspace_candidate
    
    # Method 3: Try to find workspace root by looking for marker files
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        # Look for characteristic files that indicate workspace root
        if any((parent / marker).exists() for marker in [
            'MODULE.bazel', 'WORKSPACE', 'BUILD.bazel', '.git'
        ]):
            return parent
    
    # Method 4: Default to current directory
    return current


def resolve_workspace_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a path relative to the workspace root.
    
    Args:
        relative_path: Path relative to workspace root (e.g., "data/outputs")
        
    Returns:
        Absolute path resolved from workspace root
    """
    workspace_root = get_workspace_root()
    return workspace_root / relative_path


def ensure_output_directory(relative_path: Union[str, Path]) -> Path:
    """Ensure an output directory exists, resolved from workspace root.
    
    Args:
        relative_path: Directory path relative to workspace root
        
    Returns:
        Absolute path to the created directory
    """
    output_path = resolve_workspace_path(relative_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_output_file(relative_path: Union[str, Path]) -> Path:
    """Resolve an output file path and ensure its parent directory exists.
    
    Args:
        relative_path: File path relative to workspace root
        
    Returns:
        Absolute path to the output file (directory will exist)
    """
    file_path = resolve_workspace_path(relative_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


if __name__ == "__main__":
    """Test the path resolution utilities."""
    print("🔧 Testing Bazel Path Resolution Utilities")
    print("=" * 50)
    
    workspace_root = get_workspace_root()
    print(f"Workspace root: {workspace_root}")
    
    # Test resolving various paths
    test_paths = [
        "data/outputs",
        "data/outputs/test.json",
        "configs/preprocessing_config.yaml",
        "data/CLEAR.csv"
    ]
    
    for path in test_paths:
        resolved = resolve_workspace_path(path)
        exists = resolved.exists()
        print(f"  {path} -> {resolved} {'✅' if exists else '❌'}")
    
    # Test environment variables
    print(f"\nEnvironment variables:")
    for var in ['BUILD_WORKSPACE_DIRECTORY', 'RUNFILES_DIR', 'PWD']:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")