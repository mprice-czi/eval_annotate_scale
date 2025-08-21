# Annotation at Scale

<img src="https://chanzuckerberg.com/wp-content/themes/czi/img/logo.svg" alt="CZI Logo" title="CZI Logo" style="width: 100%; height: auto;">

<br/><br/>

A production-ready annotation system for the Chan Zuckerberg Initiative's Education Evaluators team. This project provides intelligent passage preprocessing and annotation workflows for educational AI evaluation using Gemini AI, Bazel, and modern Python tooling.

## Technology Stack

- **Build System**: Bazel 8.3+ with Bzlmod for modern, reproducible builds and dependency management
- **Language**: Python 3.13.5 with strict type checking and annotations
- **Environment**: Anaconda/Conda for dependency management  
- **AI Framework**: Google Gemini AI via LangChain for intelligent text processing
- **Configuration**: YAML-based centralized configuration system
- **Code Quality**: Black formatting, MyPy type checking, and pytest testing framework

## Repository Contents

| Path | Description | Key Files |
| ---- | ----------- | --------- |
| `src/` | Main source code including CLI application and config management | [`config_manager.py`](src/config_manager.py), [`bazel_utils.py`](src/bazel_utils.py) |
| `scripts/` | Intelligent preprocessing and environment validation scripts | [`segment_passages.py`](scripts/segment_passages.py), [`generate_marginal_pairs.py`](scripts/generate_marginal_pairs.py) |
| `data/` | CLEAR corpus dataset and annotation schemas | [`CLEAR.csv`](data/CLEAR.csv), [`README.md`](data/README.md) |
| `configs/` | YAML configuration files for preprocessing pipeline | [`preprocessing_config.yaml`](configs/preprocessing_config.yaml) |
| `docs/` | Detailed guides and documentation | [`configuration_guide.md`](docs/configuration_guide.md), [`segment_passages_guide.md`](docs/segment_passages_guide.md) |

## 📚 Documentation

- **[Configuration Guide](docs/configuration_guide.md)** - Complete configuration reference for all pipeline settings
- **[Stage 1: Passage Segmentation Guide](docs/segment_passages_guide.md)** - Detailed guide for the segmentation script
- **[Stage 2: Marginal Pairs Guide](docs/generate_marginal_pairs_guide.md)** - Detailed guide for pair generation
- **[Data Directory Guide](data/README.md)** - Comprehensive data format and schema documentation
- **[Development Guide (CLAUDE.md)](CLAUDE.md)** - Complete developer reference for Claude Code users

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

# Verify CLEAR.csv dataset (should show 4,724 valid records)
bazel run //scripts:verify_clear_count

# Demo processing pipeline (no API calls, shows data structure)
bazel run //scripts:demo_processing

# Production: Two-stage robust processing pipeline
# Stage 1: Segment passages with caching and recovery
bazel run //scripts:segment_passages -- --config configs/preprocessing_config.yaml --output data/outputs/segmented_passages.json --max-passages 50 --resume

# Stage 2: Generate marginal pairs from segmented passages
bazel run //scripts:generate_marginal_pairs -- --input data/outputs/segmented_passages.json --config configs/preprocessing_config.yaml --output data/outputs/marginal_pairs.json --target-pairs 25

# Unified: Complete pipeline in single command (orchestrates both stages)
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

# Unified: Complete pipeline in single command (orchestrates both stages)
python scripts/intelligent_passage_preprocessing.py --config configs/preprocessing_config.yaml --output data/outputs/marginal_pairs.json --max-passages 50 --target-pairs 25
```

**Note**: Both approaches now work correctly thanks to workspace-aware path resolution. Output files will be created in the expected `data/outputs/` directory.

## Dataset: CLEAR Corpus

This project processes the **CLEAR (Corpus of Linguistic Educational Assessment Resources)** dataset, containing 4,724 educational text samples with comprehensive readability metrics including:

- **Text samples** from diverse educational materials and reading assessments
- **Flesch-Kincaid scores** for quantitative complexity measurement
- **Grade level indicators** and complexity categorization
- **Source document metadata** ensuring pairs come from different original texts

The dataset is carefully parsed with UTF-8-BOM handling and validated against JSON schemas for data integrity.

## Intelligent Passage Preprocessing

This project provides a robust, two-stage AI-powered preprocessing pipeline that segments CLEAR corpus passages and identifies marginally decidable pairs for vocabulary complexity annotation tasks.

### Two-Stage Pipeline (Recommended for debugging/development)

**Stage 1: Passage Segmentation (`scripts/segment_passages.py`)**
- **Intelligent Segmentation**: Uses Gemini AI to segment CLEAR corpus passages into contextually complete, readable chunks optimized for configurable reading time (default 10-15 seconds)
- **Advanced Features**: Strategic overlapping for better time compliance, vocabulary complexity assessment per segment, reading time flexibility controls
- **Caching & Recovery**: Intermediate results cached in `*_cache.json` with resume capability via `*_progress.json` state files
- **Failure Resilience**: Skip completed passages, retry failed ones, generate fallback segments for AI failures
- **Progress Tracking**: Detailed progress with batch processing, rate limiting, and comprehensive logging

**Stage 2: Marginal Pair Generation (`scripts/generate_marginal_pairs.py`)**
- **Strategic Pairing**: Configurable within-category, adjacent-category, and cross-category pairing with source document separation
- **Smart Filtering**: Multi-stage business rule filtering (Flesch score differences, reading time similarity, quality requirements) before expensive AI assessment
- **Marginality Assessment**: AI-driven evaluation to identify passage pairs that are marginally decidable for vocabulary complexity
- **Quality Scoring**: Multi-factor scoring system including confidence, Flesch diversity, vocabulary overlap, and length penalties
- **Stateless Design**: Can be re-run safely, no intermediate state to manage

### Key Advantages
- **Separation of Concerns**: Each stage has a focused responsibility with independent configuration and optimization
- **Cost Control**: Multi-stage filtering reduces expensive AI API calls by 80-90%
- **Production Ready**: Comprehensive error handling, retry logic, and resume capability for long-running jobs
- **Quality Assurance**: JSON schema validation, comprehensive logging, and intermediate result inspection
- **Scalability**: Process the full CLEAR corpus (4,724 records) with progress tracking and failure recovery
- **Workspace-Aware**: Automatic path resolution works seamlessly with both Bazel and direct Python execution
- **Configurable**: Extensive YAML configuration system for fine-tuning all aspects of processing
- **Tested & Reliable**: All scripts extensively tested and verified working correctly with real data

### Performance Expectations

- **Stage 1**: ~20-25 seconds per passage (AI segmentation)
- **Stage 2**: ~20-25 seconds per candidate pair (AI assessment)
- **Typical Workflow**: 50 passages → ~25 minutes for Stage 1, then Stage 2 time depends on pair candidates
- **Resume Capability**: Interrupted processing can be resumed from last successful passage

### Configuration System
The preprocessing pipeline uses `configs/preprocessing_config.yaml` to configure:

- **Gemini API settings** (model, temperature, retries, timeouts)
- **Segmentation parameters** (target reading time, flexibility range, overlap settings, vocabulary analysis)
- **Marginality thresholds** (confidence levels, Flesch score ranges, pair selection criteria) 
- **Pairing strategy** (within/adjacent/cross-category limits, source separation, reproducible seeding)
- **Processing limits** (API rate limiting, batch sizes, concurrent requests, cost controls)
- **Quality controls** (context preservation, vocabulary focus requirements, error exclusion)

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

### Unified Pipeline (Single Command Convenience)
The `scripts/intelligent_passage_preprocessing.py` orchestrator provides a single-command interface that runs both stages sequentially. It benefits from the same robustness features as the individual stages while offering convenience for automated workflows and one-shot processing.

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

## Support & Feedback

For questions, issues, or feedback about this educational AI annotation system, please open a GitHub issue in this repository with:

- Clear description of the problem or question
- Steps to reproduce (for bugs)
- Your environment details (Python version, OS, etc.)
- Relevant log output or error messages

## Partner with CZI

The Chan Zuckerberg Initiative is committed to advancing educational AI for the public good. If you're interested in collaborating on educational AI research or would like to learn more about our work in this space, please visit [chanzuckerberg.com](https://chanzuckerberg.com).

## Disclaimer

The resources provided in this repository are made available "as-is", without warranties or guarantees of any kind. They may contain inaccuracies, limitations, or other constraints depending on the context of use.

By accessing or using these resources, you acknowledge that:

- You are responsible for evaluating their suitability for your specific use case.
- CZI makes no representations about the accuracy, completeness, or fitness of these resources for any particular purpose.
- Any use of the materials is at your own risk, and CZI is not liable for any direct or indirect consequences that may result.
- Please consult individual documentation for resource-specific limitations and guidance.
