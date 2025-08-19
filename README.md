# Annotation at Scale

<img src="https://chanzuckerberg.com/wp-content/themes/czi/img/logo.svg" alt="CZI Logo" title="CZI Logo" style="width: 100%; height: auto;">

<br/><br/>

A production-ready annotation system for the Chan Zuckerberg Initiative's Education Evaluators team. This project provides scalable annotation workflows for educational AI evaluation using LangGraph, Bazel, and modern Python tooling.

## Technology Stack

- **Build System**: Bazel 8.3+ with Bzlmod for modern, reproducible builds and dependency management
- **Language**: Python 3.13.5 with type annotations
- **Environment**: Anaconda/Conda for dependency management
- **AI Framework**: LangGraph for stateful, multi-actor workflows
- **Testing**: Pytest with coverage reporting
- **Code Quality**: Black, Flake8, MyPy, and pre-commit hooks

## Repository Contents

| Path | Description |
| ---- | ----------- |
| `src/` | Main source code with annotation, workflows, and utilities |
| `tests/` | Comprehensive test suite |
| `data/` | Sample datasets and annotation schemas |
| `prompts/` | LangGraph prompts and templates |
| `scripts/` | Development and deployment automation scripts |
| `configs/` | Configuration files for different environments |
| `tools/` | Code formatting and linting tools |

## Quick Start

### Prerequisites
- Python 3.13.5
- Anaconda/Miniconda
- Bazel 8.3+

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd eval_annotate_scale

# Run setup script (creates conda environment and builds project)
./scripts/setup_dev.sh

# Activate the environment
conda activate eval-annotate-scale
```

### Basic Usage

```bash
# Build the entire project
bazel build //...

# Run all tests
bazel test //...

# Run the main application
bazel run //src:main -- --help

# Format code
bazel run //tools:format

# Lint code  
bazel run //tools:lint
```

# Support & Feedback
We want to hear from you. For questions or feedback, please open an issue. 

# Partner with us
If you would like to participate in our private beta or partner with us to further the public good of educational AI, [reach out to us](https://link-to-your-contact-form.com).

# Disclaimer
The resources provided in this repository are made available "as-is", without warranties or guarantees of any kind. They may contain inaccuracies, limitations, or other constraints depending on the context of use.

By accessing or using these resources, you acknowledge that:
- You are responsible for evaluating their suitability for your specific use case.
- CZI makes no representations about the accuracy, completeness, or fitness of these resources for any particular purpose.
- Any use of the materials is at your own risk, and CZI is not liable for any direct or indirect consequences that may result.
- Please consult individual documentation for resource-specific limitations and guidance.

