# Configuration Guide

This guide explains all configuration options for the preprocessing pipeline, covering both the new two-stage architecture and legacy settings.

## Configuration File Structure

The main configuration file is `configs/preprocessing_config.yaml`. It contains settings for both stages of the pipeline and can be used with legacy scripts as well.

## Configuration Sections

### Gemini API Configuration

Controls the AI model and API behavior for both stages:

```yaml
gemini:
  model: "gemini-2.5-pro"         # AI model to use (recommended: gemini-2.5-pro)
  temperature: 0.3                # Creativity/randomness (0.0-1.0, lower = more consistent)
  max_retries: 3                  # Number of retries for failed API calls
  timeout_seconds: 30             # Maximum time to wait for AI response
```

**Recommendations:**
- **Production**: Use `gemini-2.5-pro` with `temperature: 0.3` for consistency
- **Development**: Try `temperature: 0.5` for more creative segmentation
- **High-volume**: Increase `timeout_seconds` to 60 for better reliability

### Stage 1: Passage Segmentation Settings

Controls how passages are broken down into segments:

```yaml
segmentation:
  target_reading_time_seconds: 12.5    # Target reading time per segment
  target_word_count_range: [50, 150]   # Approximate word count bounds
  words_per_minute: 200               # Reading speed assumption
  preserve_context: true              # Maintain semantic coherence
  min_segment_sentences: 1            # Minimum sentences per segment
```

**Key Parameters:**

| Parameter | Purpose | Typical Values | Impact |
|-----------|---------|----------------|---------|
| `target_reading_time_seconds` | Optimal reading time | 10-15 seconds | Affects annotation task difficulty |
| `target_word_count_range` | Word count bounds | [50, 150] | Balances readability vs. context |
| `preserve_context` | Semantic coherence | `true` | Critical for quality |
| `min_segment_sentences` | Sentence completeness | 1-2 | Prevents mid-sentence breaks |

**Tuning Guidelines:**
- **Shorter segments** (8-10s): For novice annotators or complex texts
- **Longer segments** (15-20s): For expert annotators or simpler texts  
- **Narrow word range** [60, 120]: More consistent segment lengths
- **Wide word range** [40, 200]: More flexible segmentation

### Stage 2: Marginality Assessment Settings

Controls how passage pairs are identified and selected:

```yaml
marginality:
  confidence_threshold: 0.6           # Minimum AI confidence for inclusion
  flesch_difference_range: [5, 25]    # Acceptable complexity differences
  max_candidate_pairs: 100            # Limit for cost control
  target_marginal_pairs: 50           # Final output target
```

**Critical Parameters:**

| Parameter | Purpose | Recommended Range | Effect on Output |
|-----------|---------|------------------|------------------|
| `confidence_threshold` | Quality filter | 0.5-0.8 | Higher = fewer, higher-quality pairs |
| `flesch_difference_range` | Complexity gap | [5, 25] | Wider = more candidates |
| `max_candidate_pairs` | Cost control | 50-200 | Higher = better selection but more expensive |
| `target_marginal_pairs` | Output size | 25-100 | Final annotation task count |

**Optimization Strategies:**
- **High Quality**: `confidence_threshold: 0.75`, narrow Flesch range [8, 20]
- **High Volume**: `confidence_threshold: 0.55`, wide Flesch range [5, 30]
- **Cost Control**: `max_candidate_pairs: 50`, `target_marginal_pairs: 25`
- **Comprehensive**: `max_candidate_pairs: 200`, `target_marginal_pairs: 100`

### Pairing Strategy Settings

Controls the logic for creating passage pairs:

```yaml
pairing:
  within_category_pairs: 20           # Max pairs within same complexity level
  adjacent_category_pairs: 15         # Max pairs between adjacent levels
  cross_category_skip: true           # Skip extreme differences (Easy↔Hard)
  exclude_same_original: true         # CRITICAL: No pairs from same document
  random_seed: 42                     # For reproducible results
```

**Strategy Parameters:**

| Setting | Purpose | Recommended Value | Rationale |
|---------|---------|------------------|-----------|
| `within_category_pairs` | Same-level pairs | 15-25 | Tests fine-grained discrimination |
| `adjacent_category_pairs` | Cross-level pairs | 10-20 | Tests boundary recognition |
| `cross_category_skip` | Extreme differences | `true` | Avoids obvious comparisons |
| `exclude_same_original` | Source diversity | `true` | **CRITICAL** - prevents bias |

### Processing Limits (Cost & Performance)

Controls resource usage and API costs:

```yaml
limits:
  max_passages_per_batch: 50          # Batch size for Stage 1
  max_concurrent_requests: 5          # Parallel API calls
  delay_between_requests_ms: 100      # Rate limiting delay
  daily_api_call_limit: 1000          # Safety limit
```

**Performance Tuning:**

| Scenario | Batch Size | Concurrent | Delay | Daily Limit |
|----------|------------|------------|-------|-------------|
| **Development** | 10 | 2 | 200ms | 100 |
| **Production** | 20 | 3 | 100ms | 2000 |
| **High-Volume** | 50 | 5 | 50ms | 5000 |
| **Rate-Limited** | 5 | 1 | 500ms | 500 |

### Quality Control Settings

Ensures output meets quality standards:

```yaml
quality:
  require_context_preservation: true   # Only use context-preserved segments
  min_vocabulary_focus_words: 2        # Minimum complexity indicators
  exclude_segments_with_errors: true   # Skip problematic segments
```

**Quality Gates:**
- `require_context_preservation: true` - **CRITICAL** for annotation quality
- `min_vocabulary_focus_words: 2` - Ensures complexity analysis depth
- `exclude_segments_with_errors: true` - Prevents corrupted data

### Output Format Settings

Controls what information is included in output files:

```yaml
output:
  include_metadata: true              # Processing statistics and timestamps
  include_reasoning: true             # AI explanations for decisions
  sort_by_confidence: true            # Order pairs by quality score
  export_intermediate_results: false  # Save debugging information
```

## Environment-Specific Configurations

### Development Configuration
Optimized for testing and iteration:

```yaml
# configs/dev_config.yaml
gemini:
  temperature: 0.5  # More creative for testing edge cases

segmentation:
  target_reading_time_seconds: 10.0  # Shorter for faster processing

marginality:
  confidence_threshold: 0.5  # Lower bar to see more candidates
  max_candidate_pairs: 30    # Reduce costs
  target_marginal_pairs: 15  # Smaller output

limits:
  max_passages_per_batch: 10  # Smaller batches
  delay_between_requests_ms: 200  # Conservative rate limiting
  daily_api_call_limit: 200   # Development quota

output:
  export_intermediate_results: true  # Enable debugging
```

### Production Configuration
Optimized for quality and reliability:

```yaml
# configs/prod_config.yaml  
gemini:
  temperature: 0.2  # Very consistent outputs
  max_retries: 5    # More resilient to failures
  timeout_seconds: 60  # Allow longer processing

marginality:
  confidence_threshold: 0.75  # High quality threshold
  flesch_difference_range: [8, 20]  # Tighter range

pairing:
  within_category_pairs: 25    # More comprehensive coverage
  adjacent_category_pairs: 20

limits:
  max_passages_per_batch: 20   # Balance speed and reliability
  delay_between_requests_ms: 150  # Conservative rate limiting
  daily_api_call_limit: 3000   # Production quota

quality:
  min_vocabulary_focus_words: 3  # Stricter quality requirements
```

### High-Volume Configuration
Optimized for processing large datasets:

```yaml
# configs/high_volume_config.yaml
gemini:
  timeout_seconds: 45  # Allow for variable response times

segmentation:
  target_word_count_range: [60, 120]  # Tighter range for consistency

marginality:
  max_candidate_pairs: 200  # Better selection from larger pool
  target_marginal_pairs: 100  # Large output set

limits:
  max_passages_per_batch: 50  # Larger batches
  max_concurrent_requests: 5  # Maximum parallelism
  delay_between_requests_ms: 50  # Aggressive rate limiting
  daily_api_call_limit: 5000  # High quota
```

## Configuration Validation

The scripts automatically validate configuration files and report errors:

### Common Validation Errors

**Missing Required Sections:**
```
❌ Configuration missing required section: 'gemini'
✅ Add gemini section with model and temperature
```

**Invalid Value Ranges:**
```
❌ confidence_threshold must be between 0.0 and 1.0, got: 1.5
✅ Set confidence_threshold: 0.75
```

**Inconsistent Settings:**
```
❌ target_marginal_pairs (100) > max_candidate_pairs (50)
✅ Increase max_candidate_pairs or reduce target_marginal_pairs
```

## Usage Examples

### Using Different Configurations

```bash
# Development configuration
bazel run //scripts:segment_passages -- \
    --config configs/dev_config.yaml \
    --output data/outputs/dev_segments.json

# Production configuration  
bazel run //scripts:segment_passages -- \
    --config configs/prod_config.yaml \
    --output data/outputs/prod_segments.json

# Custom configuration
bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segments.json \
    --config configs/custom_pairing.yaml \
    --output data/outputs/custom_pairs.json
```

### Overriding Configuration Settings

Command line arguments override configuration file settings:

```bash
# Override target pairs regardless of config file setting
bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segments.json \
    --config configs/preprocessing_config.yaml \
    --target-pairs 75 \
    --output data/outputs/pairs.json
```

## Configuration Best Practices

### 1. Version Control Configuration Files
- Keep multiple configuration variants in version control
- Document the purpose of each configuration file
- Use meaningful names: `prod_config.yaml`, `dev_config.yaml`, `test_config.yaml`

### 2. Environment-Specific Overrides
- Use separate configs for different environments
- Never commit API keys in configuration files
- Use `.env` files or secure parameter stores for secrets

### 3. Iterative Tuning
- Start with conservative settings for new datasets
- Monitor quality metrics and adjust thresholds
- Keep logs of configuration changes and their impacts

### 4. Cost Management
- Set appropriate daily limits for API usage
- Use development configs for testing to avoid costs
- Monitor API usage through configuration telemetry

This comprehensive configuration system allows fine-tuned control over every aspect of the preprocessing pipeline while maintaining reasonable defaults for common use cases.