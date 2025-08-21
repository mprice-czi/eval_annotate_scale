#!/usr/bin/env python3
"""
Setup script for preprocessing environment.

This script helps users set up their environment for intelligent passage preprocessing
with proper security practices.
"""

import sys
from pathlib import Path

# Add src to Python path to import config_manager
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_manager import SecureConfigManager


def main():
    """Set up preprocessing environment with security best practices."""
    print("🚀 Setting up Intelligent Passage Preprocessing Environment")
    print("=" * 60)

    config_manager = SecureConfigManager()

    # Validate current environment
    print("\n🔍 Validating current environment...")
    validations = config_manager.validate_environment()

    # Provide setup guidance
    if not validations["gemini_api_key_available"]:
        print("\n🔑 API Key Setup Needed:")
        print("1. Get your Gemini API key: https://makersuite.google.com/app/apikey")
        print("2. Add to .bazelrc.local (recommended):")
        print("   run --run_env=GEMINI_API_KEY='your-api-key-here'")
        print("3. Or set environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")

    if not validations["clear_dataset_exists"]:
        print("\n📊 CLEAR Dataset Setup:")
        print("1. Ensure CLEAR.csv is in data/ directory")
        print("2. If missing, extract from CLEAR.csv.zip")

    # Step 5: Test with Bazel
    print("\n🏗️  Bazel Integration:")
    print("You can run preprocessing with Bazel for better environment isolation:")
    print("  bazel run //scripts:validate_environment")
    print(
        "  bazel run //scripts:intelligent_preprocessing -- --output data/test_output.json"
    )

    # Final status - check critical requirements only
    critical_valid = validations["gemini_api_key_available"]
    if critical_valid:
        print("\n🎉 Environment setup complete! Ready for preprocessing.")
        print("ℹ️  Optional items (config file) can be addressed later if needed.")
        return 0
    else:
        print("\n⚠️  Setup incomplete. Please address the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
