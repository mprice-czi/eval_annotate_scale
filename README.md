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

# Create conda environment and install dependencies
conda create -n eval-annotate-scale python=3.13
conda activate eval-annotate-scale
pip install -r requirements.txt

# Set up API key (one-time setup)
# Place your Gemini API key in a temporary 'key' file
python scripts/setup_bazel_env.py
rm key  # Delete temporary key file

# Build the entire project
bazel build //...

# Validate environment setup
bazel run //scripts:validate_environment
```

### Basic Usage

#### Option A: Bazel (Recommended)
```bash
# Build the project
bazel build //...

# Run the simple CLI application
bazel run //src:main -- --name "Developer"

# Validate your environment setup
bazel run //scripts:validate_environment

# Verify CLEAR.csv dataset (should show 4,724 records)
bazel run //scripts:verify_clear_count

# Demo processing pipeline (no API calls, shows data structure)
bazel run //scripts:demo_processing

# Production: Two-stage robust processing pipeline
# Stage 1: Segment passages with caching and recovery
bazel run //scripts:segment_passages -- --config configs/preprocessing_config.yaml --output data/outputs/segmented_passages.json --max-passages 50 --resume

# Stage 2: Generate marginal pairs from segmented passages
bazel run //scripts:generate_marginal_pairs -- --input data/outputs/segmented_passages.json --config configs/preprocessing_config.yaml --output data/outputs/marginal_pairs.json --target-pairs 25

# Legacy: Original monolithic preprocessing (for backwards compatibility)
bazel run //scripts:intelligent_preprocessing -- --config configs/preprocessing_config.yaml --output data/outputs/marginal_pairs.json --max-passages 50 --target-pairs 25
```

#### Option B: Direct Python Execution
```bash
# Activate conda environment first
conda activate eval-annotate-scale

# Validate your environment setup
python scripts/validate_environment.py

# Demo processing pipeline
python scripts/demo_processing.py

# Production: Two-stage processing pipeline
# Stage 1: Segment passages with caching and recovery
python scripts/segment_passages.py --config configs/preprocessing_config.yaml --output data/outputs/segmented_passages.json --max-passages 50 --resume

# Stage 2: Generate marginal pairs from segmented passages
python scripts/generate_marginal_pairs.py --input data/outputs/segmented_passages.json --config configs/preprocessing_config.yaml --output data/outputs/marginal_pairs.json --target-pairs 25
```

**Note**: Both approaches now work correctly thanks to workspace-aware path resolution. Output files will be created in the expected `data/outputs/` directory.

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
- **Workspace-Aware**: Automatic path resolution works with both Bazel and direct Python execution
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

## Development

### Code Quality Tools

```bash
# Format code (Black: 88 char line length, Python 3.13)
black src/ tools/ scripts/

# Import sorting (isort: Black-compatible profile)
isort src/ tools/ scripts/

# Type checking (MyPy: strict mode, disallow untyped defs)
mypy src/ tools/ scripts/

# Run tests (when test files are present)
pytest tests/ -v
```

### Workspace-Aware Path Resolution

This project includes intelligent path resolution that works seamlessly in both Bazel and direct Python execution environments:

- **Automatic workspace detection**: Uses `BUILD_WORKSPACE_DIRECTORY` environment variable when available (Bazel)
- **Fallback mechanisms**: Falls back to detecting workspace root via marker files (MODULE.bazel, .git)
- **Universal compatibility**: All relative paths work correctly regardless of execution method
- **Output safety**: Automatically ensures output directories exist and paths resolve correctly

The `src/bazel_utils.py` module handles this automatically - no manual configuration needed.

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

