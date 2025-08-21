#!/usr/bin/env python3
"""
Environment validation script for Bazel integration.

This script validates that the environment is properly set up for running
the intelligent passage preprocessing pipeline.
"""

import os
import sys
from pathlib import Path

# Add src to Python path to import config_manager
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_manager import SecureConfigManager


def main():
    """Validate environment and report results."""
    print("🔍 Validating Environment Setup...")
    print("=" * 50)

    # Find workspace root - try several methods
    possible_roots = [
        Path.cwd(),  # Current directory
        Path(__file__).parent.parent,  # Script's parent/parent
        Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")),  # Bazel workspace
    ]

    workspace_root = None
    for root in possible_roots:
        if (root / "MODULE.bazel").exists() or (root / "WORKSPACE").exists():
            workspace_root = root
            break

    if workspace_root:
        print(f"📁 Found workspace root: {workspace_root}")
        os.chdir(workspace_root)
    else:
        print("⚠️  Could not find workspace root, using current directory")

    # Create config manager with config file
    config_manager = SecureConfigManager("configs/preprocessing_config.yaml")

    # Run validation
    validations = config_manager.validate_environment()

    # Print results
    print("\n📊 Validation Results:")
    all_passed = True
    for check, result in validations.items():
        if check == "api_key_source":
            status = "🔑" if result != "none" else "❌"
            print(f"  {status} API Key Source: {result}")
            if result == "none":
                all_passed = False
        else:
            status = "✅" if result else "❌"
            print(f"  {status} {check.replace('_', ' ').title()}: {result}")
            if not result:
                all_passed = False

    if all_passed:
        print("\n🎉 All validations passed!")
        print("✅ Environment setup is complete and ready for preprocessing.")
        return 0
    else:
        print("\n⚠️  Some validations failed, but this may be expected:")
        print("  • config_file_loaded: Optional (preprocessing will use defaults)")
        print("  • clear_dataset_exists: Required only for actual processing")
        print("  • bazelrc_local_exists: Should be present for Bazel execution")

        # Check critical items
        critical_passed = validations.get("gemini_api_key_available", False)
        if critical_passed:
            print("\n✅ Critical requirement met: API key is available")
            print("🚀 Ready for preprocessing tasks!")
            return 0
        else:
            print("\n❌ Critical requirement missing: API key not available")
            return 1


if __name__ == "__main__":
    exit(main())
