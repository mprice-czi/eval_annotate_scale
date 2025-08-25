# Marginal Pair Generation Guide - Stage 2 Processing

> **Navigation**: [📄 Main README](../README.md) | [📊 Data Guide](../data/README.md) | [⚙️ Configuration Guide](configuration_guide.md) | [🔧 Stage 1 Guide](segment_passages_guide.md) | [👨‍💻 Development Guide](../CLAUDE.md)

This guide covers the `scripts/generate_marginal_pairs.py` script, which is Stage 2 of the robust two-stage preprocessing pipeline.

## Overview

The marginal pair generation script takes segmented passages from Stage 1 and intelligently identifies pairs that are "marginally decidable" for vocabulary complexity annotation tasks. It uses smart filtering and AI assessment to create high-quality annotation tasks.

## Features

### 🚀 Statistical Pre-filtering (NEW)
- **ML-based Selection**: scikit-learn powered intelligent pair filtering
- **85% Cost Reduction**: Dramatically fewer AI assessments needed
- **Strategic Sampling**: Cluster boundaries, marginal pairs, diversity sampling
- **Scalable Architecture**: Handles 1000+ passages efficiently

### 🎯 Smart Candidate Filtering
- **Business Rules**: Filter pairs based on Flesch score differences, complexity levels, and reading times
- **Source Separation**: Ensure pairs come from different original documents
- **Quality Gates**: Only consider passages with preserved context
- **Cost Control**: Limit expensive AI assessments to promising candidates

### 🤖 AI-Powered Assessment
- **Marginality Evaluation**: Determine if pairs are appropriately challenging but decidable
- **Quality Scoring**: Multi-factor scoring system for optimal pair selection
- **Reasoning Generation**: Detailed explanations for each pair's selection
- **Confidence Scoring**: AI confidence levels for pair appropriateness

### 🔄 Stateless Design
- **Reliable Re-execution**: Can be run multiple times safely
- **No Resume Needed**: Processes all input data in single run
- **Fast Iterations**: Quick turnaround for configuration changes
- **Memory Efficient**: Processes candidates in batches

## Usage

### Basic Usage
```bash
# Generate pairs from segmented passages
bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segmented_passages.json \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/marginal_pairs.json \
    --target-pairs 50
```

### Advanced Usage
```bash
# Custom configuration with more pairs
bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segmented_passages.json \
    --config configs/custom_pairing_config.yaml \
    --output data/outputs/high_quality_pairs.json \
    --target-pairs 100
```

## Command Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--input` | ✅ | Path to segmented passages JSON from Stage 1 | `data/outputs/segmented_passages.json` |
| `--config` | ✅ | Path to YAML configuration file | `configs/preprocessing_config.yaml` |
| `--output` | ✅ | Output path for marginal pairs JSON | `data/outputs/marginal_pairs.json` |
| `--target-pairs` | ❌ | Number of pairs to generate | `50` (overrides config) |

## Configuration

The script uses settings from `configs/preprocessing_config.yaml`:

```yaml
# Marginality Assessment Settings  
marginality:
  confidence_threshold: 0.6        # Minimum AI confidence for inclusion
  flesch_difference_range: [5, 25] # Acceptable Flesch score differences
  max_candidate_pairs: 1000        # 🚀 INCREASED: Higher limit thanks to statistical pre-filtering
  target_marginal_pairs: 50        # Final output target

# Pairing Strategy
pairing:
  within_category_pairs: 20        # Max pairs within same complexity bucket
  adjacent_category_pairs: 15      # Max pairs between adjacent levels
  cross_category_skip: true        # Skip Easy-Hard pairs (too obvious)
  exclude_same_original: true      # CRITICAL: No pairs from same original passage
  random_seed: 42                  # For reproducible sampling

# Processing Limits (API Cost Management)
limits:
  max_concurrent_requests: 5
  delay_between_requests_ms: 100
  daily_api_call_limit: 1000

# Quality Control
quality:
  require_context_preservation: true   # Only use context-preserved segments
  min_vocabulary_focus_words: 2        # Minimum complexity indicators
  exclude_segments_with_errors: true   # Skip problematic segments

# 🚀 Statistical Pre-filtering - NEW: ML-based intelligent pair selection  
statistical_filtering:
  enabled: true                        # Enable ML pre-filtering (85%+ API call reduction)
  max_tfidf_features: 1000             # TF-IDF vocabulary size for text analysis
  n_clusters: 6                        # Number of complexity clusters for strategic sampling
  pca_components: 10                   # PCA components for dimensionality reduction
  boundary_pairs_ratio: 0.33           # Portion of candidates from cluster boundaries
  marginal_pairs_ratio: 0.33           # Portion from within-cluster marginal pairs
  diverse_pairs_ratio: 0.34            # Portion from diversity sampling
```

## Candidate Filtering Logic

The script uses a two-phase approach: **Statistical Pre-filtering** → **Traditional Filtering**:

### 🚀 Phase 1: Statistical Pre-filtering (NEW)
When enabled (default), the script uses ML-based intelligent selection:

#### 1. Feature Extraction
- **TF-IDF Vectorization**: Analyze vocabulary patterns across all passages
- **Numerical Features**: Flesch scores, word counts, reading times, complexity metrics
- **Combined Features**: Merge text and numerical features for comprehensive analysis

#### 2. Clustering & Strategic Sampling
- **K-means Clustering**: Group passages by similarity (complexity, vocabulary, structure)
- **Boundary Pairs**: Sample pairs near cluster boundaries (different complexity groups)
- **Marginal Pairs**: Find subtle differences within clusters
- **Diversity Sampling**: Ensure broad coverage across the feature space

#### 3. Marginality Likelihood Scoring
- **Cosine Similarity**: Text feature comparison
- **Euclidean Distance**: Numerical feature comparison  
- **Flesch Score Optimization**: Target optimal complexity differences
- **Vocabulary Diversity**: Prefer pairs with different focus words

**Result**: ~85% reduction in candidates requiring expensive AI assessment

### Phase 2: Traditional Filtering (Applied to Pre-filtered Candidates)

#### Stage 1: Source Separation
- ✅ Only pair passages from different original documents
- ❌ Never pair subpassages from the same CLEAR record

#### Stage 2: Complexity Filtering
- ✅ Flesch score difference in configured range (5-25 points typical)
- ✅ Same or adjacent complexity levels (Easy↔Medium, Medium↔Hard)
- ❌ Skip extreme differences (Easy↔Very Hard)

#### Stage 3: Quality Filtering  
- ✅ Both passages must have `context_preserved: true`
- ✅ Similar reading times (within 10 seconds)
- ✅ Both passages have vocabulary focus words

#### Stage 4: Volume Control
- Sort candidates by statistical marginality likelihood score
- Limit to `max_candidate_pairs` for AI cost control

## AI Assessment Criteria

The AI evaluates each candidate pair for:

### Marginality Factors
- **Clear but not obvious difference**: Definitive answer exists but requires thought
- **Vocabulary-focused**: Differences primarily in word choice, not sentence structure
- **Appropriate difficulty**: Challenging but decidable in 30-60 seconds
- **Educational value**: Useful for annotator training and evaluation

### Quality Scoring System
The script calculates composite quality scores using:
- **Base Score**: AI confidence (0.6-1.0)
- **Flesch Bonus**: +0.1 for optimal score differences
- **Vocabulary Bonus**: +0.1 for diverse focus words with minimal overlap
- **Length Penalty**: -0.1 for very short/long passages outside ideal range
- **Time Bonus**: +0.05 for similar reading times

## Output Format

The script generates `marginal_pairs.json` with this structure:

```json
{
  "metadata": {
    "total_pairs": 50,
    "generation_timestamp": "2024-01-15T15:45:00Z", 
    "schema_version": "1.0",
    "average_confidence": 0.75,
    "average_quality_score": 0.82
  },
  "marginal_pairs": [
    {
      "pair_id": "pair-001",
      "passage_a": {
        "segment_id": "400-seg-1",
        "text": "When the young people returned...",
        "complexity_estimate": "Medium",
        "flesch_score": 81.7,
        "vocabulary_focus_words": ["decidedly", "interior", "landscape"]
      },
      "passage_b": {
        "segment_id": "401-seg-1", 
        "text": "Dolly looked rather wistful...",
        "complexity_estimate": "Medium",
        "flesch_score": 72.1,
        "vocabulary_focus_words": ["wistful", "uncertain", "objections"]
      },
      "marginality_confidence": 0.73,
      "complexity_difference": 9.6,
      "reasoning": "Both passages contain moderately complex vocabulary but differ in emotional versus descriptive language complexity...",
      "target_annotation_time": 45.0,
      "pair_quality_score": 0.78,
      "generation_timestamp": "2024-01-15T15:44:30Z"
    }
  ]
}
```

## Quality Metrics

The script reports key quality indicators:

### Statistical Metrics
- **Total Pairs Generated**: Final count vs. target
- **Average Confidence**: Mean AI confidence across all pairs
- **Average Quality Score**: Composite quality metric
- **Complexity Distribution**: Spread across Easy/Medium/Hard levels

### Assessment Success Rates
- **Candidate Filter Rate**: Percentage passing business rules
- **AI Assessment Rate**: Percentage surviving marginality evaluation
- **Final Selection Rate**: Percentage selected for output

## Error Handling

### Input Validation
- **File Format**: Validates JSON structure and required fields
- **Schema Compliance**: Checks segmented passages against schema
- **Content Validation**: Ensures passages have required metadata

### Processing Errors
- **API Failures**: Skip problematic pairs, continue processing others
- **Assessment Failures**: Log errors but don't stop entire process
- **Output Validation**: Verify final pairs meet all requirements

## Performance Optimization

### For Large Input Sets (1000+ passages)
- **Enable Statistical Pre-filtering**: Set `statistical_filtering.enabled: true` (default)
- **Scale ML Parameters**: Increase `n_clusters` to 8, `max_tfidf_features` to 2000
- **Higher Candidate Limits**: Use `max_candidate_pairs: 2000` (pre-filtering makes this affordable)
- **Stricter Final Selection**: Increase `confidence_threshold` to 0.75
- **Monitor Processing**: Check statistical filtering reduction ratios in logs

### For Development/Testing
- **Small Dataset Mode**: Set `n_clusters: 3`, `max_tfidf_features: 200` for <100 passages
- **Test Statistical Filtering**: Run `bazel run //scripts:test_statistical_filtering`
- **Scalability Testing**: Run `bazel run //scripts:test_scalability` 
- **Debug Mode**: Use smaller input files from Stage 1 testing
- **Relaxed Thresholds**: Lower `confidence_threshold` to see more candidates
- **Disable Pre-filtering**: Set `statistical_filtering.enabled: false` for comparison

## Integration Examples

### Full Pipeline
```bash
# Stage 1: Segment passages
bazel run //scripts:segment_passages -- \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/segments.json \
    --max-passages 100

# Stage 2: Generate pairs  
bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segments.json \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/pairs.json \
    --target-pairs 25
```

### Iterative Refinement
```bash
# Test with different thresholds
bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segments.json \
    --config configs/high_confidence_config.yaml \
    --output data/outputs/pairs_strict.json

bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segments.json \
    --config configs/moderate_confidence_config.yaml \
    --output data/outputs/pairs_moderate.json
```

## Troubleshooting

### No Pairs Generated
**Problem**: Script completes but generates zero pairs
**Solutions**:
- Check `confidence_threshold` - may be too high
- Verify input passages have `context_preserved: true`
- Review `flesch_difference_range` - may be too restrictive

### Low Quality Pairs
**Problem**: Generated pairs seem too easy or too hard
**Solutions**:
- Adjust `flesch_difference_range` to tighter bounds
- Increase `confidence_threshold` for stricter AI selection
- Review AI reasoning in output for quality insights

### API Rate Limiting
**Problem**: Many API failures due to rate limits
**Solutions**:
- Increase `delay_between_requests_ms`
- Reduce `max_concurrent_requests`
- Process in smaller batches over multiple runs

### Insufficient Candidates
**Problem**: Few candidates survive filtering
**Solutions**:
- Check that Stage 1 preserved context properly
- Verify different original documents in input
- Expand `flesch_difference_range` if too narrow

### Memory Issues
**Problem**: High memory usage with large inputs
**Solutions**:
- Process smaller input files in batches
- Reduce `max_candidate_pairs` limit
- Monitor and restart if needed

## Validation & Quality Control

### Output Validation
- JSON schema compliance checking
- Cross-reference validation (all segment IDs exist)
- Quality score distribution analysis
- Confidence score statistics

### Manual Review Process
1. **Sample Review**: Examine top 10 pairs by quality score
2. **Edge Case Analysis**: Review lowest-scoring included pairs
3. **Reasoning Evaluation**: Assess AI reasoning quality
4. **Annotation Testing**: Trial annotation of sample pairs

This systematic approach ensures high-quality marginal pairs suitable for productive annotation tasks.

---

## Related Documentation

- [📄 Main README](../README.md) - Project overview and quick start
- [📊 Data Directory Guide](../data/README.md) - Complete data format reference
- [⚙️ Configuration Guide](configuration_guide.md) - Complete configuration reference
- [🔧 Stage 1: Passage Segmentation Guide](segment_passages_guide.md) - Previous stage in the pipeline
- [👨‍💻 Development Guide (CLAUDE.md)](../CLAUDE.md) - Complete developer reference