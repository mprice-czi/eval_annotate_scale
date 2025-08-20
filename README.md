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

# NEW: Two-stage robust processing pipeline (Recommended)
# Stage 1: Segment passages with caching and recovery
bazel run //scripts:segment_passages -- --config configs/preprocessing_config.yaml --output data/outputs/segmented_passages.json --max-passages 50 --resume

# Stage 2: Generate marginal pairs from segmented passages
bazel run //scripts:generate_marginal_pairs -- --input data/outputs/segmented_passages.json --config configs/preprocessing_config.yaml --output data/outputs/marginal_pairs.json --target-pairs 25

# Legacy: Original monolithic preprocessing (for backwards compatibility)
bazel run //scripts:intelligent_preprocessing -- --config configs/preprocessing_config.yaml --output data/outputs/marginal_pairs.json --max-passages 50 --target-pairs 25
```

## Intelligent Passage Preprocessing

This project provides a robust, two-stage AI-powered preprocessing pipeline that segments text passages and identifies marginally decidable pairs for vocabulary complexity annotation tasks.

### New Two-Stage Architecture (Recommended)

**Stage 1: Passage Segmentation (`scripts/segment_passages.py`)**
- **Intelligent Segmentation**: Uses Gemini AI to segment CLEAR corpus passages into contextually complete, readable chunks optimized for 10-15 second reading time
- **Caching & Recovery**: Intermediate results cached with resume capability for long-running jobs
- **Failure Resilience**: Skip completed passages, retry failed ones, fallback segments for AI failures
- **Progress Tracking**: Detailed progress with batch processing and rate limiting

**Stage 2: Marginal Pair Generation (`scripts/generate_marginal_pairs.py`)**
- **Smart Filtering**: Business rule-based candidate filtering before expensive AI assessment
- **Marginality Assessment**: AI-driven evaluation to identify passage pairs that are marginally decidable for vocabulary complexity
- **Quality Scoring**: Multi-factor quality assessment for optimal pair selection
- **Stateless Design**: Can be re-run safely, no intermediate state to manage

### Key Advantages
- **Separation of Concerns**: Each stage has a focused responsibility
- **Cost Control**: Only assess marginality on pre-filtered candidates 
- **Production Ready**: Handle failures gracefully with resume capability
- **Debugging**: Intermediate results are cached and inspectable
- **Scalability**: Process thousands of passages without losing work
- **Tested & Reliable**: All scripts tested and verified working correctly

### Performance Expectations
- **Stage 1**: ~20-25 seconds per passage (AI segmentation)
- **Stage 2**: ~20-25 seconds per candidate pair (AI assessment)
- **Typical Workflow**: 50 passages → ~25 minutes for Stage 1, then Stage 2 time depends on pair candidates
- **Resume Capability**: Interrupted processing can be resumed from last successful passage

### Configuration System
The preprocessing pipeline uses `configs/preprocessing_config.yaml` to configure:
- **Gemini API settings** (model, temperature, retries, timeouts)
- **Segmentation parameters** (target reading time, word count ranges, batch sizes)
- **Marginality thresholds** (confidence levels, pair selection criteria)
- **Processing limits** (API rate limiting, concurrent requests, cost controls)
- **Quality controls** (context preservation, vocabulary requirements)

### Pipeline Outputs
**Stage 1 Output** (`data/outputs/segmented_passages.json`):
- Segmented passages with complexity estimates
- Vocabulary focus words and reading time estimates
- Processing metadata and source text hashes
- Context preservation and quality flags

**Stage 2 Output** (`data/outputs/marginal_pairs.json`):
- Marginally decidable passage pairs with confidence scores
- Quality scores and marginality reasoning
- Processing statistics and metadata
- AI reasoning explanations for each pair

### Legacy Pipeline
The original monolithic `scripts/intelligent_preprocessing.py` remains available for backwards compatibility but is not recommended for production use due to lack of failure recovery.

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

