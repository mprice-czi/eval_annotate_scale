# Development Scripts

This directory contains helper scripts for the annotation at scale project.

## Available Scripts

### Environment Management
- `setup_dev.sh` - Set up development environment with conda and dependencies
- `clean_env.sh` - Clean up development environment

### Development Workflow  
- `run_tests.sh` - Run the full test suite with coverage
- `format_code.sh` - Format all Python code using black and isort
- `lint_code.sh` - Lint code using flake8 and mypy
- `build_all.sh` - Build all Bazel targets

### Data Processing
- `process_annotations.py` - Process annotation files and generate reports
- `validate_data.py` - Validate input data schemas

### Deployment
- `deploy_dev.sh` - Deploy to development environment
- `docker_build.sh` - Build Docker images for containerized deployment