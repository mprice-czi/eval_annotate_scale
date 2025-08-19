#!/usr/bin/env python3
"""
Secure Configuration Manager

Handles secure retrieval of API keys and sensitive configuration from environment variables,
following security best practices for the eval_annotate_scale project.

Never stores secrets in code, version control, or plain text files.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

class SecureConfigManager:
    """Secure configuration manager for API keys and sensitive settings."""
    
    # Supported environment variable names for API keys
    GEMINI_API_KEY_VARS = [
        'GEMINI_API_KEY',
        'GOOGLE_API_KEY', 
        'GOOGLE_AI_API_KEY',
        'GOOGLE_GENERATIVE_AI_API_KEY'
    ]
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to non-sensitive config file (YAML)
        """
        self.config_file = config_file
        self._config_cache: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load non-sensitive configuration from file."""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    self._config_cache = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
                self._config_cache = {}
        else:
            self._config_cache = {}
    
    def get_gemini_api_key(self) -> Optional[str]:
        """Securely retrieve Gemini API key from environment variables.
        
        Returns:
            API key if found in environment, None otherwise
        """
        for var_name in self.GEMINI_API_KEY_VARS:
            api_key = os.getenv(var_name)
            if api_key:
                logger.info(f"Found Gemini API key in environment variable: {var_name}")
                return api_key.strip()
        
        logger.warning("No Gemini API key found in environment variables")
        return None
    
    def require_gemini_api_key(self) -> str:
        """Require Gemini API key, exit with error message if not found.
        
        Returns:
            API key from environment
            
        Raises:
            SystemExit: If API key not found
        """
        api_key = self.get_gemini_api_key()
        if not api_key:
            self._print_api_key_setup_instructions()
            raise SystemExit("❌ Required Gemini API key not found in environment")
        return api_key
    
    def _print_api_key_setup_instructions(self) -> None:
        """Print helpful instructions for setting up API key."""
        print("\n🔑 API Key Setup Required")
        print("=" * 50)
        print("To use Gemini AI features, set your API key as an environment variable.")
        
        print("\nRECOMMENDED - Use Bazel configuration:")
        print("1. Run setup script to create .bazelrc.local:")
        print("   python scripts/setup_bazel_env.py")
        print("2. Then use Bazel commands:")
        print("   bazel run //scripts:intelligent_preprocessing -- --output data/output.json")
        
        print("\nAlternative - Direct environment variable:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        
        print("\nAlternative - Project .env file:")
        print("  cp .env.example .env")
        print("  # Edit .env file and add your key")
        print("  source .env")
        
        print("\n📍 Get your API key at: https://makersuite.google.com/app/apikey")
        print("\n🚨 SECURITY REMINDERS:")
        print("  • Use .bazelrc.local for secure Bazel integration (git-ignored)")
        print("  • NEVER commit API keys to version control!")
        print("  • NEVER put API keys in code or command line arguments!")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get non-sensitive configuration value.
        
        Args:
            key: Configuration key (supports dot notation like 'gemini.temperature')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config_cache
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate that required environment setup is correct.
        
        Returns:
            Dictionary of validation results
        """
        api_key = self.get_gemini_api_key()
        
        # Check what method provided the API key
        api_key_source = "none"
        if api_key:
            for var_name in self.GEMINI_API_KEY_VARS:
                if os.getenv(var_name):
                    api_key_source = f"env_var_{var_name}"
                    break
        
        # Try multiple possible paths for files (depending on execution context)
        possible_roots = [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent]
        
        def find_file(relative_path: str) -> bool:
            """Try to find a file in various possible root directories."""
            for root in possible_roots:
                if (root / relative_path).exists():
                    return True
            return False
        
        validations = {
            'gemini_api_key_available': api_key is not None,
            'api_key_source': api_key_source,
            'config_file_loaded': bool(self._config_cache),
            'clear_dataset_exists': find_file('data/CLEAR.csv'),
            'bazelrc_local_exists': find_file('.bazelrc.local'),
        }
        
        # Log validation results
        for check, result in validations.items():
            if check == 'api_key_source':
                status = "🔑" if result != "none" else "❌"
                logger.info(f"{status} {check}: {result}")
            else:
                status = "✅" if result else "❌"
                logger.info(f"{status} {check}: {result}")
        
        return validations
    
    def create_env_template(self, output_path: str = '.env.example') -> None:
        """Create a template .env file for users.
        
        Args:
            output_path: Where to create the template file
        """
        template_content = '''# Environment Variables for eval_annotate_scale
# Copy this file to .env and fill in your actual values
# NEVER commit .env files with real secrets to version control!

# Gemini API Key (get from: https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=your-gemini-api-key-here

# Alternative names also supported:
# GOOGLE_API_KEY=your-api-key-here
# GOOGLE_AI_API_KEY=your-api-key-here

# Optional: Set logging level
LOG_LEVEL=INFO

# Optional: Override default data paths
# CLEAR_CSV_PATH=data/CLEAR.csv
# OUTPUT_PATH=data/preprocessed_output.json
'''
        
        with open(output_path, 'w') as f:
            f.write(template_content)
        
        logger.info(f"Created environment template at {output_path}")


# Global configuration instance
config_manager = SecureConfigManager('configs/preprocessing_config.yaml')


def get_gemini_api_key() -> str:
    """Convenience function to get Gemini API key."""
    return config_manager.require_gemini_api_key()


def validate_environment() -> bool:
    """Convenience function to validate environment setup."""
    validations = config_manager.validate_environment()
    all_valid = all(validations.values())
    
    if not all_valid:
        print("\n🚨 Environment Setup Issues Detected:")
        for check, result in validations.items():
            if not result:
                print(f"  ❌ {check}")
        print("\nPlease fix the issues above before proceeding.")
    
    return all_valid


if __name__ == "__main__":
    """Run environment validation when called directly."""
    print("🔍 Validating Environment Setup...")
    
    # Create .env template if it doesn't exist
    if not Path('.env.example').exists():
        config_manager.create_env_template()
    
    # Run validation
    is_valid = validate_environment()
    
    if is_valid:
        print("\n✅ Environment setup is valid!")
    else:
        print("\n❌ Environment setup needs attention.")
        exit(1)