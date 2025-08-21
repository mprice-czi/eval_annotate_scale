# Passage Segmentation Guide - Stage 1 Processing

> **Navigation**: [📄 Main README](../README.md) | [📊 Data Guide](../data/README.md) | [⚙️ Configuration Guide](configuration_guide.md) | [🎯 Stage 2 Guide](generate_marginal_pairs_guide.md) | [👨‍💻 Development Guide](../CLAUDE.md)

This guide covers the `scripts/segment_passages.py` script, which is Stage 1 of the robust two-stage preprocessing pipeline.

## Overview

The passage segmentation script uses Gemini AI to intelligently break down CLEAR corpus passages into contextually complete, readable chunks optimized for vocabulary complexity annotation tasks.

## Features

### 🔄 Resume Capability
- **Progress Tracking**: Maintains state in `*_progress.json` files
- **Caching**: Stores completed segments in `*_cache.json` files  
- **Smart Resume**: Skip already processed passages with `--resume` flag
- **Failure Recovery**: Handle API failures gracefully with fallback segments

### 🎯 AI-Powered Segmentation
- **Context Preservation**: Maintains semantic coherence across segment boundaries
- **Reading Time Optimization**: Target 10-15 second reading chunks
- **Vocabulary Focus**: Identifies key complexity-indicating words
- **Quality Assessment**: Evaluates context preservation and segment quality

### 🛡️ Production Reliability
- **Rate Limiting**: Configurable delays between API calls
- **Batch Processing**: Process in manageable chunks
- **Error Handling**: Comprehensive logging and fallback strategies
- **Resource Management**: Memory-efficient processing of large datasets

## Usage

### Basic Usage
```bash
# Run with default settings
bazel run //scripts:segment_passages -- \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/segmented_passages.json \
    --max-passages 100

# Resume interrupted processing
bazel run //scripts:segment_passages -- \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/segmented_passages.json \
    --resume
```

### Advanced Usage
```bash
# Process subset with custom CSV path
bazel run //scripts:segment_passages -- \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/test_segments.json \
    --clear-csv data/CLEAR_subset.csv \
    --max-passages 50 \
    --resume
```

## Command Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--config` | ✅ | Path to YAML configuration file | `configs/preprocessing_config.yaml` |
| `--output` | ✅ | Output path for segmented passages JSON | `data/outputs/segmented_passages.json` |
| `--max-passages` | ❌ | Maximum passages to process (for testing) | `100` |
| `--resume` | ❌ | Resume from previous run | Flag only |
| `--clear-csv` | ❌ | Path to CLEAR.csv file | `data/CLEAR.csv` (default) |

## Configuration

The script uses settings from `configs/preprocessing_config.yaml`:

```yaml
# Gemini API Configuration
gemini:
  model: "gemini-2.5-pro"
  temperature: 0.3
  max_retries: 3
  timeout_seconds: 30

# Segmentation Settings
segmentation:
  target_reading_time_seconds: 12.5
  target_word_count_range: [50, 150]
  preserve_context: true
  min_segment_sentences: 1

# Processing Limits
limits:
  max_passages_per_batch: 10
  delay_between_requests_ms: 100
  max_concurrent_requests: 5
```

## Output Format

The script generates `segmented_passages.json` with this structure:

```json
{
  "metadata": {
    "total_passages": 1250,
    "generation_timestamp": "2024-01-15T14:30:00Z",
    "schema_version": "1.0"
  },
  "segmented_passages": [
    {
      "original_id": "400",
      "segment_id": "400-seg-1",
      "text": "When the young people returned to the ballroom...",
      "estimated_reading_time": 12.8,
      "flesch_score": 81.7,
      "complexity_estimate": "Medium",
      "context_preserved": true,
      "vocabulary_focus_words": ["decidedly", "interior", "landscape"],
      "processing_timestamp": "2024-01-15T14:25:30Z",
      "source_text_hash": "a1b2c3d4"
    }
  ]
}
```

## Intermediate Files

During processing, the script creates temporary files:

### Progress File (`*_progress.json`)
```json
{
  "completed_ids": ["400", "401", "402"],
  "failed_ids": ["403"],
  "last_processed_index": 25,
  "total_processed": 28,
  "start_time": 1705329000.123
}
```

### Cache File (`*_cache.json`)
Contains the same format as the final output but updated incrementally.

## Error Handling & Recovery

### Common Scenarios

**API Rate Limits**
- Automatic retry with exponential backoff
- Configurable delay between requests
- Graceful degradation to fallback segments

**Network Failures**
- Resume from last successful passage
- Preserve all completed work in cache
- Detailed error logging for debugging

**AI Processing Failures**
- Generate fallback segments from original text
- Mark segments with `context_preserved: false`
- Continue processing remaining passages

### Recovery Commands

```bash
# Resume after failure
bazel run //scripts:segment_passages -- \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/segmented_passages.json \
    --resume

# Clean restart (removes cache/progress files)
bazel run //scripts:segment_passages -- \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/segmented_passages.json
    # No --resume flag
```

## Performance Optimization

### For Large Datasets
- Use `max_passages_per_batch: 5` for better memory management
- Increase `delay_between_requests_ms` to avoid rate limits
- Monitor cache file growth and disk space

### For Development/Testing
- Set `max_passages: 20` for quick iterations
- Use higher `temperature: 0.5` for more creative segmentation
- Enable debug logging in configuration

## Validation

The script validates output against the JSON schema:
- Schema file: `data/schemas/processed_passage.json`
- Validates structure, types, and required fields
- Reports validation errors with detailed messages

## Troubleshooting

### Script Won't Resume
**Problem**: Resume flag ignored, processing restarts from beginning
**Solution**: Check that output directory contains `*_progress.json` and `*_cache.json` files

### High Memory Usage  
**Problem**: Script consumes excessive memory with large datasets
**Solution**: Reduce `max_passages_per_batch` in configuration

### API Quota Exceeded
**Problem**: Rate limit errors from Gemini API
**Solution**: Increase `delay_between_requests_ms` or reduce `max_concurrent_requests`

### Invalid Segments Generated
**Problem**: AI generates malformed or incomplete segments  
**Solution**: Check `fallback_segments` in logs, adjust AI prompt parameters

### Progress File Corruption
**Problem**: Resume fails due to corrupted progress file
**Solution**: Delete `*_progress.json` and `*_cache.json`, restart without resume

## Integration with Stage 2

The output from this script serves as input to Stage 2:

```bash
# Stage 1: Generate segments
bazel run //scripts:segment_passages -- \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/segmented_passages.json

# Stage 2: Generate pairs from segments  
bazel run //scripts:generate_marginal_pairs -- \
    --input data/outputs/segmented_passages.json \
    --config configs/preprocessing_config.yaml \
    --output data/outputs/marginal_pairs.json
```

This separation allows for:
- Independent optimization of each stage
- Easier debugging of pipeline issues
- Flexible processing of different datasets
- Better resource utilization and cost control

---

## Related Documentation

- [📄 Main README](../README.md) - Project overview and quick start
- [📊 Data Directory Guide](../data/README.md) - Complete data format reference
- [⚙️ Configuration Guide](configuration_guide.md) - Complete configuration reference
- [🎯 Stage 2: Marginal Pairs Guide](generate_marginal_pairs_guide.md) - Next stage in the pipeline
- [👨‍💻 Development Guide (CLAUDE.md)](../CLAUDE.md) - Complete developer reference