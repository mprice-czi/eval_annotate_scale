#!/usr/bin/env python3
"""
Setup Bazel Environment Configuration

This script reads the API key from the 'key' file and creates a proper
Bazel environment configuration in .bazelrc.local (git-ignored).

Usage:
    python scripts/setup_bazel_env.py

This will:
1. Read API key from 'key' file
2. Create .bazelrc.local with proper environment variables
3. The 'key' file can then be safely deleted
"""

import os
import sys
from pathlib import Path

def read_key_file() -> str:
    """Read API key from the 'key' file."""
    key_file = Path('key')
    
    if not key_file.exists():
        print("❌ Error: 'key' file not found in project root")
        print("Please ensure the API key file exists before running this script")
        sys.exit(1)
    
    try:
        with open(key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        
        if len(api_key) < 20:
            print(f"⚠️  Warning: API key seems short ({len(api_key)} characters)")
            print("Please verify the key file contains a valid API key")
        
        print(f"✅ Successfully read API key from 'key' file ({len(api_key)} characters)")
        return api_key
        
    except Exception as e:
        print(f"❌ Error reading 'key' file: {e}")
        sys.exit(1)

def create_bazelrc_local(api_key: str) -> None:
    """Create .bazelrc.local with environment configuration."""
    
    bazelrc_content = f'''# Bazel Local Configuration (Auto-generated)
# This file contains environment variables for API keys
# DO NOT COMMIT THIS FILE - it's in .gitignore

# API Key Environment Variables for AI Processing
build --action_env=GEMINI_API_KEY="{api_key}"
build --action_env=GOOGLE_API_KEY="{api_key}"
build --action_env=GOOGLE_AI_API_KEY="{api_key}"

# Also set for test runs
test --action_env=GEMINI_API_KEY="{api_key}"
test --action_env=GOOGLE_API_KEY="{api_key}"
test --action_env=GOOGLE_AI_API_KEY="{api_key}"

# Run environment (for py_binary targets) - use --run_env for runtime
run --run_env=GEMINI_API_KEY="{api_key}"
run --run_env=GOOGLE_API_KEY="{api_key}"
run --run_env=GOOGLE_AI_API_KEY="{api_key}"
# Also keep action_env for build-time dependencies
run --action_env=GEMINI_API_KEY="{api_key}"
run --action_env=GOOGLE_API_KEY="{api_key}"
run --action_env=GOOGLE_AI_API_KEY="{api_key}"
'''
    
    bazelrc_local = Path('.bazelrc.local')
    
    try:
        with open(bazelrc_local, 'w', encoding='utf-8') as f:
            f.write(bazelrc_content)
        
        print(f"✅ Created {bazelrc_local} with environment configuration")
        print("   This file is git-ignored and contains your API key securely")
        
    except Exception as e:
        print(f"❌ Error creating .bazelrc.local: {e}")
        sys.exit(1)

def verify_bazelrc_import() -> None:
    """Verify that .bazelrc imports .bazelrc.local"""
    bazelrc = Path('.bazelrc')
    
    if not bazelrc.exists():
        print("⚠️  .bazelrc file not found")
        return
    
    try:
        with open(bazelrc, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it already imports .bazelrc.local
        if '.bazelrc.local' in content or 'try-import %workspace%/.bazelrc.local' in content:
            print("✅ .bazelrc already imports .bazelrc.local")
        else:
            print("ℹ️  .bazelrc doesn't import .bazelrc.local yet")
            print("   Adding import statement...")
            
            # Add import to .bazelrc
            with open(bazelrc, 'a', encoding='utf-8') as f:
                f.write('\n# Import local configuration (git-ignored)\n')
                f.write('try-import %workspace%/.bazelrc.local\n')
            
            print("✅ Added .bazelrc.local import to .bazelrc")
    
    except Exception as e:
        print(f"⚠️  Could not verify .bazelrc: {e}")

def main():
    """Main setup function."""
    print("🔧 Setting up Bazel Environment Configuration")
    print("=" * 50)
    
    # Step 1: Read API key from file
    print("\n1️⃣ Reading API key from 'key' file...")
    api_key = read_key_file()
    
    # Step 2: Create .bazelrc.local
    print("\n2️⃣ Creating Bazel local configuration...")
    create_bazelrc_local(api_key)
    
    # Step 3: Verify .bazelrc imports
    print("\n3️⃣ Verifying .bazelrc configuration...")
    verify_bazelrc_import()
    
    # Step 4: Instructions
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. The 'key' file can now be safely deleted:")
    print("   rm key")
    print("\n2. Your API key is now securely configured in .bazelrc.local")
    print("   (This file is git-ignored)")
    print("\n3. Run Bazel commands normally:")
    print("   bazel run //scripts:intelligent_preprocessing -- --output data/output.json")
    print("\n4. Validate the setup:")
    print("   bazel run //scripts:validate_environment")
    
    print("\n🔐 Security: .bazelrc.local is git-ignored and contains your API key")

if __name__ == "__main__":
    main()