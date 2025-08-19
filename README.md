# Annotation at Scale

<img src="https://chanzuckerberg.com/wp-content/themes/czi/img/logo.svg" alt="CZI Logo" title="CZI Logo" style="width: 100%; height: auto;">

<br/><br/>

A production-ready annotation system for the Chan Zuckerberg Initiative's Education Evaluators team. This project provides intelligent passage preprocessing and annotation workflows for educational AI evaluation using LangGraph, Bazel, and modern Python tooling.

## Technology Stack

- **Build System**: Bazel 8.3+ with Bzlmod for modern, reproducible builds and dependency management
- **Language**: Python 3.13.5 with type annotations
- **Environment**: Anaconda/Conda for dependency management
- **AI Framework**: LangGraph for stateful, multi-actor workflows
- **Configuration**: YAML-based configuration system
- **Code Quality**: PEP 8 compliance with structured imports and best practices

## Repository Contents

| Path | Description |
| ---- | ----------- |
| `src/` | Main source code including CLI application and config management |
| `scripts/` | Intelligent preprocessing and environment validation scripts |
| `data/` | CLEAR corpus dataset and annotation schemas |
| `configs/` | YAML configuration files for preprocessing pipeline |
| `tools/` | Development utilities and helper scripts |

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

# Create conda environment
conda env create -f environment.yml
conda activate eval-annotate-scale

# Build the entire project
bazel build //...
```

### Basic Usage

```bash
# Build the project
bazel build //...

# Run the simple CLI application
bazel run //src:main -- --name "Developer"

# Set up Bazel environment with API key (one-time setup)
bazel run //scripts:setup_bazel_env

# Validate your environment setup
bazel run //scripts:validate_environment

# Run intelligent passage preprocessing with configuration
bazel run //scripts:intelligent_preprocessing -- --config configs/preprocessing_config.yaml --output data/marginal_pairs.json

# Alternative: Use custom configuration
bazel run //scripts:intelligent_preprocessing -- --config my_custom_config.yaml --max-passages 50 --target-pairs 25
```

## Intelligent Passage Preprocessing

This project provides an AI-powered preprocessing pipeline that intelligently segments text passages and identifies marginally decidable pairs for vocabulary complexity annotation tasks.

### Core Features
- **Intelligent Segmentation**: Uses Gemini AI to segment CLEAR corpus passages into contextually complete, readable chunks optimized for 10-15 second reading time
- **Marginality Assessment**: AI-driven evaluation to identify passage pairs that are marginally decidable for vocabulary complexity
- **LangGraph Workflows**: Stateful, multi-step processing with proper error handling and batching
- **YAML Configuration**: Comprehensive configuration system for all pipeline parameters

### Configuration System
The preprocessing pipeline uses `configs/preprocessing_config.yaml` to configure:
- **Gemini API settings** (model, temperature, retries)
- **Segmentation parameters** (target reading time, word count ranges)
- **Marginality thresholds** (confidence levels, pair selection criteria)
- **Processing limits** (batch sizes, API rate limiting)
- **Quality controls** (context preservation, vocabulary requirements)

### Pipeline Output
The system generates JSON files containing:
- Segmented passages with complexity estimates
- Marginally decidable passage pairs with confidence scores
- Metadata including processing parameters and statistics
- Reasoning explanations for AI decisions

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

