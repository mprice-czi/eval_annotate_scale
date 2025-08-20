# CLEAR Corpus Data Directory

This directory contains the complete data infrastructure for the "Annotation @ Scale (Vocabulary)" experiment, including the CLEAR corpus dataset, JSON validation schemas, and comprehensive examples. This document provides a complete, self-contained guide to understanding, accessing, and working with all data components.

## Table of Contents

- [Directory Structure](#directory-structure)
- [CLEAR Corpus Dataset](#clear-corpus-dataset)
- [JSON Schema System](#json-schema-system)
- [Sample Data and Examples](#sample-data-and-examples)
- [Data Processing Workflows](#data-processing-workflows)
- [CSV Parsing Guide](#csv-parsing-guide)
- [Validation Procedures](#validation-procedures)
- [Integration Examples](#integration-examples)
- [Troubleshooting](#troubleshooting)

## Directory Structure

```
data/
├── CLEAR.csv                           # Primary dataset (6.1MB, 4,724 records)
├── README.md                           # This comprehensive documentation
├── BUILD.bazel                         # Bazel build targets and exports
├── schemas/                            # JSON Schema validation definitions
│   ├── clear_record.json               # Schema for raw CLEAR corpus records
│   ├── processed_passage.json          # Schema for AI-processed passage segments
│   └── marginal_pair.json              # Schema for annotation task pairs
├── examples/                           # Sample data demonstrating all formats
│   ├── clear_records_sample.json       # Representative CLEAR corpus records
│   ├── processed_passages_sample.json  # AI-processed passage examples
│   └── marginal_pairs_sample.json      # Annotation pair examples
├── fixtures/                           # Test data for automated testing (planned)
└── outputs/                            # Generated processing results (gitignored)
    ├── segmented_passages.json          # Stage 1 output: segmented passages
    ├── segmented_passages_cache.json    # Stage 1 cache for resume capability
    ├── segmented_passages_progress.json # Stage 1 progress tracking
    ├── marginal_pairs.json              # Stage 2 output: final marginal pairs
    └── *.log                            # Processing log files
```

## CLEAR Corpus Dataset

### Overview and Context

The **CLEAR** (Corpus of Linguistic and Educational Analysis Resources) dataset serves as the foundation for vocabulary complexity annotation experiments. This corpus was specifically chosen for its diverse range of text complexity levels, comprehensive readability metrics, and educational relevance.

### Dataset Specifications

- **Filename**: `CLEAR.csv`
- **File Size**: 6.1MB (uncompressed)
- **Total Records**: 4,724 text samples
- **Source Format**: CSV with complex multi-line header structure
- **Encoding**: UTF-8 with BOM (Byte Order Mark)
- **Origin**: Primarily Project Gutenberg literature collection
- **Publication Range**: Early 20th century to modern texts

### Critical CSV Structure Details

The CLEAR.csv file has a **non-standard CSV structure** that requires careful handling:

**Important**: The CSV file contains 4,726 total rows, but 2 rows at the end are empty/invalid, resulting in 4,724 valid records with complete data.

#### Header Structure (Line 1)
The CSV header is contained in a single line with column names containing embedded newlines within quoted fields. The complete header must be processed as a unit to correctly identify the 40 data columns.

```
Line 1: ﻿ID,Last Changed,Author,Title,Anthology,URL,Source,Pub Year,Category,Location,License,"MPAA
Max","MPAA 
#Max","MPAA
#Avg",Excerpt,"Google
WC","Joon
WC v1",British WC,British Words,"Sentence
Count v1","Sentence
Count v2",Paragraphs,[...continuing with readability metrics and predictions]
```

#### Data Records (Lines 2+)
Actual data begins at line 2. Each record contains 40 fields, with several critical formatting considerations:

1. **Multi-line Text Fields**: The "Excerpt" field (column 15) contains multi-paragraph text with embedded newlines
2. **Quoted Fields**: Text fields are properly quoted to handle commas and newlines
3. **Nullable Fields**: Some fields (Last Changed, Anthology, British Words) may be empty
4. **Numeric Precision**: Readability scores and predictions use high-precision decimal values

### Complete Column Reference

| Column # | Field Name | Data Type | Description | Example Values |
|----------|------------|-----------|-------------|----------------|
| 1 | ID | Integer | Unique identifier | 400, 401, 402 |
| 2 | Last Changed | String/Null | Modification timestamp | Usually empty |
| 3 | Author | String | Text author name | "Carolyn Wells", "Mark Twain" |
| 4 | Title | String | Work title | "Patty's Suitors" |
| 5 | Anthology | String/Null | Collection name | Usually empty |
| 6 | URL | String | Source URL | Project Gutenberg URLs |
| 7 | Source | String | Data source | "gutenberg" |
| 8 | Pub Year | Integer | Publication year | 1914, 1917, 1885 |
| 9 | Category | String | Text category | "Lit" (Literature) |
| 10 | Location | String | Geographic context | "mid", "brit", "am" |
| 11 | License | String | Usage rights | "PD" (Public Domain), "PG" |
| 12 | MPAA Max | String | Content rating | "G", "PG", "PG-13" |
| 13 | MPAA #Max | Integer | Rating count | 1, 2, 3 |
| 14 | MPAA #Avg | Float | Average rating | 1.0, 1.5, 2.0 |
| 15 | Excerpt | String | Text sample | Multi-paragraph literary text |
| 16 | Google WC | Integer | Google word count | 164, 174, 189 |
| 17 | Joon WC v1 | Integer | Alternative word count | 184, 179, 195 |
| 18 | British WC | Integer | British word count | 0, 1, 2 |
| 19 | British Words | String/Null | British words found | "traveller", "colour" |
| 20 | Sentence Count v1 | Integer | Sentence count (method 1) | 11, 15, 8 |
| 21 | Sentence Count v2 | Integer | Sentence count (method 2) | 11, 15, 8 |
| 22 | Paragraphs | Integer | Paragraph count | 6, 4, 3 |
| 23 | BT Easiness | Float | Bradley-Terry easiness | -0.340259125 |
| 24 | BT s.e. | Float | Bradley-Terry std error | 0.464009046 |
| 25 | Flesch-Reading-Ease | Float | Flesch Reading Ease | 81.7, 80.26 |
| 26 | Flesch-Kincaid-Grade-Level | Float | Grade level | 5.95, 4.86 |
| 27 | Automated Readability Index | Float | ARI score | 7.37, 4.16 |
| 28 | SMOG Readability | Integer | SMOG score | 8, 7 |
| 29 | New Dale-Chall Readability Formula | Float | Dale-Chall score | 6.55, 6.25 |
| 30 | CAREC | Float | CAREC complexity | 0.12102 |
| 31 | CAREC_M | Float | CAREC modified | 0.11952 |
| 32 | CARES | Float | CARES complexity | 0.457533524 |
| 33 | CML2RI | Float | CML2RI index | 12.0978155 |
| 34 | firstPlace_pred | Float | ML prediction rank 1 | -0.383830529 |
| 35 | secondPlace_pred | Float | ML prediction rank 2 | -0.283603798 |
| 36 | thirdPlace_pred | Float | ML prediction rank 3 | -0.34687853 |
| 37 | fourthPlace_pred | Float | ML prediction rank 4 | -0.281620144 |
| 38 | fifthPlace_pred | Float | ML prediction rank 5 | -0.247767173 |
| 39 | sixthPlace_pred | Float | ML prediction rank 6 | -0.28994507 |
| 40 | Kaggle split | String | Dataset split | "Train", "Test" |

### Data Quality and Characteristics

#### Content Distribution
- **Training Set**: ~3,781 records (Kaggle split = "Train")  
- **Test Set**: ~945 records (Kaggle split = "Test")
- **Text Length**: Excerpts range from ~50 to ~300 words
- **Complexity Range**: Flesch scores from ~20 (very difficult) to ~90+ (very easy)
- **Time Periods**: 19th-21st century literature with publication years 1800-2020+

#### Quality Considerations
- **Multi-line Text**: Excerpts preserve original paragraph structure with embedded newlines
- **Character Encoding**: UTF-8 with BOM; may require encoding specification during parsing
- **Numeric Precision**: Readability metrics use up to 9 decimal places for precision
- **Missing Values**: Some fields intentionally null (Last Changed, Anthology, British Words when count=0)

## JSON Schema System

The schema system provides comprehensive validation and documentation for all data formats used in the annotation pipeline. Each schema is designed to be both human-readable and machine-enforceable.

### Schema Architecture Principles

1. **Strict Validation**: All schemas use `"additionalProperties": false` to prevent data contamination
2. **Comprehensive Types**: Every field includes detailed type information and constraints
3. **Documentation**: Each property includes detailed descriptions and examples
4. **Interoperability**: Schemas reference each other using JSON Schema `$ref` syntax
5. **Validation Ready**: Compatible with standard JSON Schema validators (Draft 07)

### CLEAR Record Schema (`schemas/clear_record.json`)

This schema validates raw CLEAR corpus records after CSV-to-JSON conversion.

#### Purpose and Usage
- **Primary Use**: Validate CLEAR records converted from CSV format
- **Validation Tool**: Ensure data integrity during ETL processes
- **Development Aid**: Provide IDE autocomplete and type checking
- **Documentation**: Serve as authoritative field reference

#### Key Schema Features
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CLEAR Corpus Record",
  "description": "Schema for individual records in the CLEAR dataset",
  "type": "object",
  "properties": {
    "id": {
      "type": "integer",
      "description": "Unique identifier for the text sample"
    },
    "excerpt": {
      "type": "string", 
      "description": "Text excerpt for analysis (may contain multiple paragraphs)"
    },
    // ... all 40 fields with types and descriptions
  },
  "required": ["id", "author", "title", "source", "excerpt", "kaggle_split"],
  "additionalProperties": false
}
```

#### Field Type Mapping
- **Integers**: ID, publication year, word counts, sentence counts, paragraph counts, SMOG score
- **Numbers (Float)**: All readability metrics, predictions, confidence scores
- **Strings**: Text fields, categorical fields, URLs
- **Nullable Fields**: `["string", "null"]` for optional fields like Last Changed, Anthology
- **Enums**: Constrained values like Source ("gutenberg"), License ("PD", "PG"), Kaggle split ("Train", "Test")

#### Validation Example
```python
import json
import jsonschema

# Load schema
with open('schemas/clear_record.json', 'r') as f:
    schema = json.load(f)

# Validate record
record = {
    "id": 400,
    "author": "Carolyn Wells", 
    "title": "Patty's Suitors",
    "source": "gutenberg",
    "excerpt": "When the young people returned...",
    "kaggle_split": "Train"
    # ... other required fields
}

try:
    jsonschema.validate(record, schema)
    print("✅ Record is valid")
except jsonschema.exceptions.ValidationError as e:
    print(f"❌ Validation error: {e.message}")
```

### Processed Passage Schema (`schemas/processed_passage.json`)

This schema validates AI-processed passage segments created by the intelligent preprocessing pipeline.

#### Purpose and Context
Processed passages are created by:
1. **Segmentation**: Breaking CLEAR excerpts into optimal reading chunks (10-15 seconds)
2. **AI Analysis**: Using Gemini AI to assess complexity and identify vocabulary focus words
3. **Metadata Enhancement**: Adding reading time estimates and context preservation flags

#### Schema Structure Deep Dive

```json
{
  "properties": {
    "original_id": {
      "type": "string",
      "description": "Reference to the original CLEAR record ID"
    },
    "segment_id": {
      "type": "string", 
      "description": "Unique identifier for this processed segment",
      "pattern": "^[0-9]+-seg-[0-9]+$"  // Format: "400-seg-1"
    },
    "text": {
      "type": "string",
      "minLength": 10,
      "description": "The processed text segment"
    },
    "estimated_reading_time": {
      "type": "number",
      "minimum": 5.0,
      "maximum": 30.0,
      "description": "Estimated reading time in seconds (target: 10-15s)"
    },
    "complexity_estimate": {
      "type": "string",
      "enum": ["Easy", "Medium", "Hard", "Very Hard"],
      "description": "AI-generated complexity assessment"
    },
    "vocabulary_focus_words": {
      "type": "array",
      "items": {"type": "string", "minLength": 2},
      "minItems": 1,
      "maxItems": 10,
      "description": "Key vocabulary words identified for complexity assessment"
    }
  }
}
```

#### Processing Workflow Integration
1. **Input**: Raw CLEAR record with multi-paragraph excerpt
2. **Segmentation**: AI identifies natural break points while preserving context
3. **Analysis**: Each segment analyzed for complexity and vocabulary difficulty
4. **Output**: Multiple processed passages per original record, each validating against this schema

#### Example Usage in Pipeline
```python
def create_processed_passage(original_record, segment_text, segment_number):
    """Create a processed passage that validates against the schema."""
    
    passage = {
        "original_id": str(original_record["id"]),
        "segment_id": f"{original_record['id']}-seg-{segment_number}",
        "text": segment_text,
        "estimated_reading_time": estimate_reading_time(segment_text),
        "flesch_score": calculate_flesch_score(segment_text),
        "complexity_estimate": ai_complexity_assessment(segment_text),
        "context_preserved": check_context_preservation(segment_text),
        "vocabulary_focus_words": extract_focus_words(segment_text)
    }
    
    # Validate against schema before returning
    validate_processed_passage(passage)
    return passage
```

### Marginal Pair Schema (`schemas/marginal_pair.json`)

This schema validates passage pairs that are "marginally decidable" for vocabulary complexity annotation tasks.

#### Theoretical Foundation
Marginal pairs are selected based on:
- **Moderate Difficulty Difference**: Complex enough to have a clear answer, simple enough to be decidable
- **Vocabulary Focus**: Differences primarily in vocabulary complexity rather than sentence structure
- **Annotation Feasibility**: Pairs that human annotators can reasonably assess in 30 seconds

#### Schema Design Philosophy
The schema uses JSON Schema `$ref` to define reusable components, ensuring consistency between passage_a and passage_b while maintaining validation strictness.

```json
{
  "definitions": {
    "PassageReference": {
      "type": "object",
      "properties": {
        "segment_id": {"type": "string"},
        "text": {"type": "string", "minLength": 10},
        "complexity_estimate": {
          "type": "string",
          "enum": ["Easy", "Medium", "Hard", "Very Hard"]
        },
        "flesch_score": {"type": "number", "minimum": 0, "maximum": 100},
        "vocabulary_focus_words": {
          "type": "array",
          "items": {"type": "string"},
          "minItems": 1
        }
      },
      "required": ["segment_id", "text", "complexity_estimate", "flesch_score", "vocabulary_focus_words"]
    }
  },
  "properties": {
    "pair_id": {
      "type": "string",
      "pattern": "^pair-[0-9]+$",
      "description": "Unique identifier for this passage pair"
    },
    "passage_a": {"$ref": "#/definitions/PassageReference"},
    "passage_b": {"$ref": "#/definitions/PassageReference"},
    "marginality_confidence": {
      "type": "number",
      "minimum": 0.5,
      "maximum": 1.0,
      "description": "AI confidence that this pair is marginally decidable (0.5-1.0)"
    },
    "complexity_difference": {
      "type": "number",
      "minimum": 1.0,
      "maximum": 30.0,
      "description": "Estimated vocabulary complexity difference (Flesch score delta)"
    }
  }
}
```

#### Marginal Pair Selection Algorithm
1. **Candidate Generation**: Create all possible pairs from processed passages
2. **Filtering**: Remove pairs with extreme complexity differences (too easy/hard to decide)
3. **AI Assessment**: Use LangGraph agents to evaluate marginality and provide reasoning
4. **Validation**: Ensure all selected pairs conform to schema requirements
5. **Quality Control**: Human review of AI reasoning for top candidates

## Sample Data and Examples

The `examples/` directory contains carefully curated sample data that demonstrates every aspect of the data pipeline. These examples are designed to be both educational and functional for development purposes.

### CLEAR Records Sample (`examples/clear_records_sample.json`)

#### Purpose and Educational Value
This file contains two complete CLEAR corpus records converted from CSV to JSON format, demonstrating:
- **Field Mapping**: How CSV columns map to JSON properties
- **Data Type Handling**: Conversion of strings to appropriate types (integers, floats, nulls)
- **Multi-line Text Preservation**: How paragraph structure is maintained in JSON
- **Null Handling**: Proper representation of empty CSV fields

#### Content Analysis

**Record 1: Carolyn Wells - "Patty's Suitors" (1914)**
```json
{
  "id": 400,
  "author": "Carolyn Wells",
  "title": "Patty's Suitors", 
  "pub_year": 1914,
  "excerpt": "When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.\nThe floor was covered with snow-white canvas...",
  "flesch_reading_ease": 81.7,
  "complexity_estimate": "Easy-Medium"
}
```

**Key Educational Points**:
- **High Flesch Score (81.7)**: Indicates relatively easy reading level
- **Descriptive Vocabulary**: Contains spatial/visual terms ("decidedly", "interior", "landscape")
- **Multi-paragraph Structure**: Excerpt preserved with `\n` newline characters
- **Historical Context**: Early 20th century literature with period-appropriate language

**Record 2: Carolyn Wells - "Two Little Women on a Holiday" (1917)**
- **Moderate Flesch Score (80.26)**: Slightly more complex than Record 1
- **Emotional Vocabulary**: Contains psychological terms ("wistful", "uncertain", "objections")
- **Dialog Integration**: Includes quoted speech with emotional context
- **Character Development Focus**: Emphasis on internal states and relationships

#### Development Usage Patterns
```python
# Load and examine sample records
with open('examples/clear_records_sample.json', 'r') as f:
    sample_records = json.load(f)

for record in sample_records:
    print(f"Record {record['id']}: {record['title']} by {record['author']}")
    print(f"Complexity: Flesch={record['flesch_reading_ease']}")
    print(f"Excerpt length: {len(record['excerpt'])} characters")
    print(f"Word count methods: Google={record['google_wc']}, Joon={record['joon_wc_v1']}")
    print("---")
```

### Processed Passages Sample (`examples/processed_passages_sample.json`)

#### AI Processing Demonstration
This file shows the results of intelligent passage preprocessing, demonstrating:
1. **Segmentation Strategy**: How longer excerpts are broken into optimal reading chunks
2. **AI Analysis Results**: Complexity estimates and vocabulary focus word identification  
3. **Metadata Enhancement**: Reading time estimates and context preservation assessment

#### Detailed Example Analysis

**Passage 1: "400-seg-1" - Descriptive Scene Setting**
```json
{
  "original_id": "400",
  "segment_id": "400-seg-1", 
  "text": "When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape. The floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field.",
  "estimated_reading_time": 12.8,
  "complexity_estimate": "Medium",
  "vocabulary_focus_words": ["decidedly", "interior", "landscape", "rumpled", "hillocks"]
}
```

**Analysis Points**:
- **Optimal Length**: 12.8-second reading time fits target range (10-15 seconds)
- **Contextual Completeness**: Segment ends at natural break (complete scene description)
- **Vocabulary Complexity**: Focus words include moderately complex terms
- **Descriptive Nature**: Spatial and visual vocabulary predominates

**Passage 2: "401-seg-1" - Emotional Character Development**
- **Emotional Vocabulary**: "wistful", "uncertain", "objections" indicate psychological complexity
- **Longer Reading Time**: 14.5 seconds due to more complex sentence structure
- **Context Preservation**: Complete thought unit about character motivation
- **Complexity Assessment**: AI identified emotional vocabulary as complexity driver

#### Processing Pipeline Integration
```python
def validate_processed_passages(passages):
    """Validate a batch of processed passages against the schema."""
    schema = load_schema('schemas/processed_passage.json')
    valid_passages = []
    
    for passage in passages:
        try:
            jsonschema.validate(passage, schema)
            # Additional business logic validation
            if 5 <= passage['estimated_reading_time'] <= 30:
                if len(passage['vocabulary_focus_words']) >= 2:
                    valid_passages.append(passage)
                else:
                    logger.warning(f"Insufficient focus words: {passage['segment_id']}")
            else:
                logger.warning(f"Reading time out of range: {passage['segment_id']}")
        except ValidationError as e:
            logger.error(f"Schema validation failed: {passage['segment_id']} - {e}")
    
    return valid_passages
```

### Marginal Pairs Sample (`examples/marginal_pairs_sample.json`)

#### Annotation Task Preparation
This file demonstrates the final output format for SuperAnnotate annotation tasks, showing:
- **Pair Selection Logic**: Why these specific passages were chosen as marginally decidable
- **Confidence Scoring**: How AI assesses the appropriateness of each pair
- **Reasoning Documentation**: Detailed explanations for human annotator guidance

#### Example Pair Analysis

**Pair 1: Descriptive vs. Emotional Vocabulary**
```json
{
  "pair_id": "pair-001",
  "marginality_confidence": 0.73,
  "complexity_difference": 9.6,
  "reasoning": "Both passages contain moderately complex vocabulary but differ in emotional versus descriptive language complexity. Passage A uses spatial/visual terms while Passage B uses psychological/emotional terminology. The Flesch score difference suggests marginal decidability for vocabulary complexity assessment."
}
```

**Educational Analysis**:
- **Moderate Confidence (0.73)**: AI is reasonably confident this pair is appropriately challenging
- **Significant but Not Extreme Difference (9.6)**: Large enough to have a clear answer, small enough to require careful consideration
- **Domain Differentiation**: Spatial/visual vs. emotional vocabulary creates interesting comparison
- **Pedagogical Value**: Annotators must distinguish between different types of complexity

#### Annotation Workflow Integration
```python
def prepare_annotation_batch(marginal_pairs, batch_size=50):
    """Prepare a batch of marginal pairs for SuperAnnotate upload."""
    
    # Sort by confidence score (highest first for quality)
    sorted_pairs = sorted(marginal_pairs, 
                         key=lambda x: x['marginality_confidence'], 
                         reverse=True)
    
    annotation_batch = []
    for pair in sorted_pairs[:batch_size]:
        annotation_task = {
            "task_id": pair["pair_id"],
            "passage_a": pair["passage_a"]["text"],
            "passage_b": pair["passage_b"]["text"],
            "complexity_hint_a": pair["passage_a"]["complexity_estimate"], 
            "complexity_hint_b": pair["passage_b"]["complexity_estimate"],
            "expected_annotation_time": pair["target_annotation_time"],
            "ai_reasoning": pair["reasoning"]  # For annotator guidance
        }
        annotation_batch.append(annotation_task)
    
    return annotation_batch
```

## Data Processing Workflows

### End-to-End Pipeline Overview

The complete data processing pipeline transforms raw CLEAR CSV data into annotation-ready passage pairs through multiple validated stages:

**New Two-Stage Pipeline (Recommended):**
```
CLEAR.csv → Stage 1: Passage Segmentation → Stage 2: Marginal Pair Generation → SuperAnnotate Tasks
```

**Legacy Monolithic Pipeline:**
```
CLEAR.csv → JSON Records → Processed Passages → Marginal Pairs → SuperAnnotate Tasks
```

### New Two-Stage Pipeline Architecture

#### Stage 1: Passage Segmentation (`scripts/segment_passages.py`)
**Input**: Raw CLEAR.csv  
**Output**: `data/outputs/segmented_passages.json` with processed passage segments  
**Key Features**:
- AI-powered segmentation using Gemini 2.5 Pro
- Caching system with `*_cache.json` and `*_progress.json` files
- Resume capability for interrupted processing
- Fallback handling for AI failures
- Batch processing with rate limiting

**Data Flow**:
```
CLEAR.csv → Load & Parse → AI Segmentation → Cached Results → segmented_passages.json
```

#### Stage 2: Marginal Pair Generation (`scripts/generate_marginal_pairs.py`)
**Input**: `data/outputs/segmented_passages.json`  
**Output**: `data/outputs/marginal_pairs.json` with marginally decidable pairs  
**Key Features**:
- Business rule-based candidate filtering
- AI-powered marginality assessment
- Multi-factor quality scoring
- Stateless design for reliable re-execution

**Data Flow**:
```
segmented_passages.json → Filter Candidates → AI Assessment → Quality Scoring → marginal_pairs.json
```

### Legacy Pipeline Stages

#### CSV to JSON Conversion (Legacy)
**Input**: Raw CLEAR.csv with complex header structure  
**Output**: Validated JSON records conforming to `clear_record.json` schema  
**Key Processes**:
- Multi-line header parsing
- Data type conversion (strings → integers/floats)
- Null value handling
- UTF-8/BOM encoding management

**Implementation Guide**:
```python
import csv
import json
from io import StringIO

def parse_clear_csv(csv_path):
    """Parse CLEAR.csv handling complex header and multi-line fields."""
    
    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
    
    # Split into header (lines 1-8) and data (lines 9+)
    lines = content.split('\n')
    header_lines = lines[:8]
    data_lines = lines[8:]
    
    # Reconstruct header by joining continuation lines
    header = ' '.join(header_lines).replace('\n', ' ')
    columns = [col.strip('"') for col in csv.reader([header]).__next__()]
    
    # Parse data with proper CSV handling for quoted multi-line fields
    csv_reader = csv.reader(data_lines)
    records = []
    
    for row in csv_reader:
        if len(row) == len(columns):  # Valid record
            record = {}
            for i, value in enumerate(row):
                column = columns[i].lower().replace(' ', '_').replace('-', '_')
                # Type conversion logic
                record[column] = convert_field_type(value, column)
            records.append(record)
    
    return records
```

#### Stage 2: Intelligent Passage Processing
**Input**: Validated CLEAR JSON records  
**Output**: Processed passage segments conforming to `processed_passage.json` schema  
**Key Processes**:
- AI-driven segmentation using Gemini
- Reading time estimation
- Complexity assessment
- Vocabulary focus word extraction

**Processing Configuration** (from `configs/preprocessing_config.yaml`):
```yaml
segmentation:
  target_reading_time_seconds: 12.5  # 10-15 second range midpoint
  target_word_count_range: [50, 150]
  preserve_context: true
  min_segment_sentences: 1
```

#### Stage 3: Marginal Pair Generation
**Input**: Collection of processed passages  
**Output**: Marginal pairs conforming to `marginal_pair.json` schema  
**Key Processes**:
- Candidate pair generation (combinations)
- Marginality assessment using LangGraph
- Confidence scoring and reasoning
- Quality filtering and ranking

### Workflow Automation Scripts

#### Complete Processing Script
```python
#!/usr/bin/env python3
"""Complete CLEAR data processing workflow with validation."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.config_manager import get_gemini_api_key
from scripts.intelligent_passage_preprocessing import process_clear_passages

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Execute complete data processing workflow."""
    
    # Stage 1: Load and validate CLEAR records
    logger.info("Stage 1: Loading CLEAR records")
    clear_records = load_clear_records('data/CLEAR.csv')
    validate_records(clear_records, 'schemas/clear_record.json')
    logger.info(f"Loaded {len(clear_records)} valid CLEAR records")
    
    # Stage 2: Process passages with AI
    logger.info("Stage 2: Processing passages with Gemini AI")  
    api_key = get_gemini_api_key()
    processed_passages = process_clear_passages(
        clear_records[:100],  # Process first 100 for testing
        config_path='configs/preprocessing_config.yaml'
    )
    validate_passages(processed_passages, 'schemas/processed_passage.json')
    logger.info(f"Generated {len(processed_passages)} processed passages")
    
    # Stage 3: Generate marginal pairs
    logger.info("Stage 3: Generating marginal pairs")
    marginal_pairs = generate_marginal_pairs(processed_passages)
    validate_pairs(marginal_pairs, 'schemas/marginal_pair.json')
    logger.info(f"Generated {len(marginal_pairs)} marginal pairs")
    
    # Save results with validation
    save_with_validation(processed_passages, 'data/outputs/processed_passages.json')
    save_with_validation(marginal_pairs, 'data/outputs/marginal_pairs.json')
    
    logger.info("✅ Complete workflow finished successfully")

if __name__ == "__main__":
    main()
```

## CSV Parsing Guide

### Critical Parsing Considerations

The CLEAR.csv format presents several parsing challenges that must be handled correctly to avoid data corruption:

#### 1. Byte Order Mark (BOM) Handling
```python
# ❌ Incorrect - may include BOM in first field
with open('data/CLEAR.csv', 'r', encoding='utf-8') as file:
    content = file.read()

# ✅ Correct - automatically handles BOM
with open('data/CLEAR.csv', 'r', encoding='utf-8-sig') as file:
    content = file.read()
```

#### 2. Multi-line Header Processing
```python
def extract_column_names(csv_path):
    """Extract column names from header with embedded newlines."""
    
    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        header_line = file.readline().strip()
    
    # Parse header as CSV row to handle quoted fields with embedded newlines
    header_reader = csv.reader([header_line])
    columns = next(header_reader)
    
    # Clean column names
    cleaned_columns = []
    for col in columns:
        # Remove quotes and normalize whitespace
        clean_col = col.strip('"').replace('\n', ' ').replace('  ', ' ')
        cleaned_columns.append(clean_col)
    
    return cleaned_columns
```

#### 3. Multi-line Field Handling
```python
def parse_data_with_multiline_fields(csv_path):
    """Parse CSV data handling quoted multi-line fields correctly."""
    
    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        # Skip header
        file.readline()
        
        # Use csv.reader to handle quoted multi-line fields
        csv_reader = csv.reader(file, quoting=csv.QUOTE_MINIMAL)
        records = []
        
        for row_num, row in enumerate(csv_reader, start=2):
            if len(row) == 40:  # Expected column count
                records.append(row)
            else:
                logger.warning(f"Line {row_num}: Expected 40 fields, got {len(row)}")
        
        return records
```

#### 4. Data Type Conversion
```python
def convert_clear_field(value: str, field_name: str) -> Any:
    """Convert CSV string values to appropriate Python types."""
    
    # Handle empty/null values
    if not value or value.strip() == '':
        return None
    
    # Integer fields
    if field_name in ['id', 'pub_year', 'google_wc', 'joon_wc_v1', 'british_wc',
                     'sentence_count_v1', 'sentence_count_v2', 'paragraphs', 
                     'mpaa_max_count', 'smog_readability']:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Could not convert '{value}' to int for {field_name}")
            return None
    
    # Float fields  
    if field_name in ['mpaa_avg', 'bt_easiness', 'bt_se', 'flesch_reading_ease',
                     'flesch_kincaid_grade_level', 'automated_readability_index',
                     'new_dale_chall', 'carec', 'carec_m', 'cares', 'cml2ri'] + \
                     [f'{pos}_place_pred' for pos in ['first', 'second', 'third', 
                      'fourth', 'fifth', 'sixth']]:
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Could not convert '{value}' to float for {field_name}")
            return None
    
    # String fields (default)
    return value.strip()
```

### Complete Parsing Implementation

```python
class CLEARParser:
    """Robust parser for CLEAR.csv with full error handling and validation."""
    
    def __init__(self, csv_path: str, schema_path: str):
        self.csv_path = Path(csv_path)
        self.schema = self.load_schema(schema_path)
        self.column_mapping = self.create_column_mapping()
    
    def parse(self) -> List[Dict[str, Any]]:
        """Parse CLEAR.csv and return validated records."""
        
        try:
            # Extract column names from header
            columns = self.extract_column_names()
            logger.info(f"Extracted {len(columns)} column names from header")
            
            # Parse data rows
            raw_records = self.parse_data_rows()
            logger.info(f"Parsed {len(raw_records)} raw data rows")
            
            # Convert to structured records
            structured_records = []
            for i, row in enumerate(raw_records):
                try:
                    record = self.convert_row_to_record(row, columns)
                    self.validate_record(record)
                    structured_records.append(record)
                except Exception as e:
                    logger.error(f"Error processing row {i+2}: {e}")
                    continue
            
            logger.info(f"Successfully converted {len(structured_records)} valid records")
            return structured_records
            
        except Exception as e:
            logger.error(f"Fatal parsing error: {e}")
            raise
    
    def validate_record(self, record: Dict[str, Any]) -> None:
        """Validate record against JSON schema."""
        try:
            jsonschema.validate(record, self.schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Record validation failed: {e.message}")
```

## Validation Procedures

### Multi-Level Validation Strategy

The data pipeline employs multiple validation layers to ensure data quality and consistency:

#### Level 1: Schema Validation
**Purpose**: Ensure structural correctness and type safety  
**Implementation**: JSON Schema validation for all data formats  
**Validation Points**: After CSV parsing, after AI processing, before annotation export

```python
import jsonschema
from pathlib import Path

class DataValidator:
    """Comprehensive data validation with detailed error reporting."""
    
    def __init__(self, schemas_dir: str = 'data/schemas'):
        self.schemas = self.load_all_schemas(schemas_dir)
    
    def validate_clear_records(self, records: List[Dict]) -> ValidationResult:
        """Validate CLEAR records with detailed error reporting."""
        
        schema = self.schemas['clear_record']
        results = ValidationResult()
        
        for i, record in enumerate(records):
            try:
                jsonschema.validate(record, schema)
                results.add_success(f"clear_record_{i}")
            except jsonschema.ValidationError as e:
                results.add_error(f"clear_record_{i}", e.message, e.json_path)
        
        return results
    
    def validate_processed_passages(self, passages: List[Dict]) -> ValidationResult:
        """Validate processed passages with business logic checks."""
        
        schema = self.schemas['processed_passage']
        results = ValidationResult()
        
        for passage in passages:
            # Schema validation
            try:
                jsonschema.validate(passage, schema)
            except jsonschema.ValidationError as e:
                results.add_error(passage['segment_id'], e.message)
                continue
            
            # Business logic validation
            if not (10 <= passage['estimated_reading_time'] <= 15):
                results.add_warning(passage['segment_id'], 
                    f"Reading time {passage['estimated_reading_time']}s outside target range")
            
            if len(passage['vocabulary_focus_words']) < 3:
                results.add_warning(passage['segment_id'], 
                    "Fewer than 3 vocabulary focus words identified")
            
            results.add_success(passage['segment_id'])
        
        return results
```

#### Level 2: Business Logic Validation
**Purpose**: Enforce domain-specific rules and quality standards  
**Examples**: Reading time ranges, complexity score bounds, vocabulary word counts

```python
def validate_business_rules(data: Dict, data_type: str) -> List[str]:
    """Validate domain-specific business rules."""
    
    errors = []
    
    if data_type == 'processed_passage':
        # Reading time validation
        reading_time = data.get('estimated_reading_time', 0)
        if not (5 <= reading_time <= 30):
            errors.append(f"Reading time {reading_time}s outside acceptable range (5-30s)")
        
        # Vocabulary focus validation
        focus_words = data.get('vocabulary_focus_words', [])
        if len(focus_words) < 2:
            errors.append("Insufficient vocabulary focus words (minimum 2 required)")
        
        # Complexity consistency
        flesch_score = data.get('flesch_score', 0)
        complexity = data.get('complexity_estimate', '')
        if not complexity_flesch_consistent(flesch_score, complexity):
            errors.append(f"Complexity estimate '{complexity}' inconsistent with Flesch score {flesch_score}")
    
    elif data_type == 'marginal_pair':
        # Marginality validation  
        confidence = data.get('marginality_confidence', 0)
        if confidence < 0.6:
            errors.append(f"Marginality confidence {confidence} below threshold (0.6)")
        
        # Complexity difference validation
        diff = data.get('complexity_difference', 0)
        if not (2 <= diff <= 25):
            errors.append(f"Complexity difference {diff} outside optimal range (2-25)")
    
    return errors
```

#### Level 3: Cross-Reference Validation
**Purpose**: Ensure data consistency across related records  
**Examples**: Passage segment IDs reference valid CLEAR records, pair passages exist in processed collection

```python
def validate_cross_references(clear_records: List[Dict], 
                            processed_passages: List[Dict],
                            marginal_pairs: List[Dict]) -> ValidationReport:
    """Validate cross-references between data collections."""
    
    report = ValidationReport()
    
    # Build reference indexes
    clear_ids = {str(record['id']) for record in clear_records}
    passage_ids = {passage['segment_id'] for passage in processed_passages}
    
    # Validate processed passage references
    for passage in processed_passages:
        original_id = passage['original_id']
        if original_id not in clear_ids:
            report.add_error(f"Processed passage {passage['segment_id']} references non-existent CLEAR record {original_id}")
    
    # Validate marginal pair references  
    for pair in marginal_pairs:
        passage_a_id = pair['passage_a']['segment_id']
        passage_b_id = pair['passage_b']['segment_id']
        
        if passage_a_id not in passage_ids:
            report.add_error(f"Marginal pair {pair['pair_id']} references non-existent passage A: {passage_a_id}")
        
        if passage_b_id not in passage_ids:
            report.add_error(f"Marginal pair {pair['pair_id']} references non-existent passage B: {passage_b_id}")
    
    return report
```

### Validation Reporting and Error Handling

```python
@dataclass
class ValidationResult:
    """Comprehensive validation result with categorized issues."""
    
    successes: List[str] = field(default_factory=list)
    warnings: List[Tuple[str, str]] = field(default_factory=list) 
    errors: List[Tuple[str, str]] = field(default_factory=list)
    
    def add_success(self, item_id: str):
        self.successes.append(item_id)
    
    def add_warning(self, item_id: str, message: str):
        self.warnings.append((item_id, message))
    
    def add_error(self, item_id: str, message: str):
        self.errors.append((item_id, message))
    
    def print_summary(self):
        print(f"✅ Successful validations: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.warnings:
            print("\nWarnings:")
            for item_id, message in self.warnings:
                print(f"  {item_id}: {message}")
        
        if self.errors:
            print("\nErrors:")
            for item_id, message in self.errors:
                print(f"  {item_id}: {message}")
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
```

## Integration Examples

### Python Integration with Pandas

```python
import pandas as pd
import json
from typing import List, Dict

def load_clear_as_dataframe() -> pd.DataFrame:
    """Load CLEAR records as pandas DataFrame for analysis."""
    
    # Method 1: Direct CSV loading (handles complex header automatically)
    df = pd.read_csv('data/CLEAR.csv', encoding='utf-8-sig')
    
    # Method 2: JSON-validated loading (recommended for production)
    with open('data/examples/clear_records_sample.json', 'r') as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    
    return df

def analyze_complexity_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze complexity score distributions."""
    
    analysis = {
        'flesch_mean': df['flesch_reading_ease'].mean(),
        'flesch_std': df['flesch_reading_ease'].std(),
        'grade_level_mean': df['flesch_kincaid_grade_level'].mean(),
        'word_count_mean': df['google_wc'].mean(),
        'complexity_by_category': df.groupby('category')['flesch_reading_ease'].mean().to_dict()
    }
    
    return analysis
```

### LangGraph Workflow Integration

```python
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

def create_passage_processing_workflow():
    """Create LangGraph workflow for passage processing."""
    
    # Define workflow state
    class PassageProcessingState(TypedDict):
        clear_record: Dict[str, Any]
        segments: List[str]
        processed_passages: List[Dict[str, Any]]
        errors: List[str]
    
    # Define workflow steps
    def segment_passage(state: PassageProcessingState) -> PassageProcessingState:
        """Segment passage into readable chunks."""
        # Implementation using Gemini AI
        pass
    
    def analyze_complexity(state: PassageProcessingState) -> PassageProcessingState:
        """Analyze complexity and extract focus words."""
        # Implementation using Gemini AI
        pass
    
    def validate_results(state: PassageProcessingState) -> PassageProcessingState:
        """Validate processed passages against schema."""
        # Implementation using JSON Schema validation
        pass
    
    # Build workflow graph
    workflow = StateGraph(PassageProcessingState)
    workflow.add_node("segment", segment_passage)
    workflow.add_node("analyze", analyze_complexity) 
    workflow.add_node("validate", validate_results)
    
    workflow.add_edge("segment", "analyze")
    workflow.add_edge("analyze", "validate")
    workflow.add_edge("validate", END)
    
    workflow.set_entry_point("segment")
    
    return workflow.compile()
```

### Bazel Integration

```python
# BUILD.bazel integration for data processing
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "data_validation",
    srcs = ["validation.py"],
    deps = [
        "//third_party:jsonschema",
        "//third_party:pandas",
    ],
    data = [
        "//data:schemas",
        "//data:examples",
    ],
)

py_binary(
    name = "process_clear_data",
    srcs = ["process_data.py"],
    main = "process_data.py",
    deps = [
        ":data_validation",
        "//src:config_manager",
    ],
    data = [
        "//data:CLEAR.csv",
        "//configs:preprocessing_config.yaml",
    ],
)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: CSV Parsing Errors
**Symptoms**: Incorrect field counts, malformed records, encoding errors  
**Causes**: Improper handling of multi-line header or BOM  
**Solutions**:
```python
# ✅ Correct approach
with open('data/CLEAR.csv', 'r', encoding='utf-8-sig') as f:
    # Skip header
    f.readline()
    
    # Use csv.reader for proper quoted field handling
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 40:  # Expected field count
            process_row(row)
```

#### Issue 2: Schema Validation Failures  
**Symptoms**: ValidationError exceptions, type conversion errors  
**Causes**: Incorrect data types, missing required fields, additional properties  
**Solutions**:
```python
# Enable detailed validation error reporting
try:
    jsonschema.validate(record, schema)
except jsonschema.ValidationError as e:
    print(f"Validation failed at path: {e.json_path}")
    print(f"Error message: {e.message}")
    print(f"Invalid value: {e.instance}")
    print(f"Schema requirement: {e.schema}")
```

#### Issue 3: Memory Usage with Large Datasets
**Symptoms**: High memory consumption, slow processing  
**Causes**: Loading entire dataset into memory  
**Solutions**:
```python
def process_clear_in_batches(csv_path: str, batch_size: int = 1000):
    """Process CLEAR data in batches to manage memory usage."""
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        # Skip header
        f.readline()
        
        reader = csv.reader(f)
        batch = []
        
        for row in reader:
            batch.append(row)
            
            if len(batch) == batch_size:
                yield process_batch(batch)
                batch = []
        
        # Process remaining records
        if batch:
            yield process_batch(batch)
```

#### Issue 4: API Rate Limiting in AI Processing  
**Symptoms**: Rate limit errors, failed API calls  
**Causes**: Too many concurrent requests to Gemini API  
**Solutions**: Configure proper rate limiting in `configs/preprocessing_config.yaml`
```yaml
limits:
  max_concurrent_requests: 5
  delay_between_requests_ms: 100
  daily_api_call_limit: 1000
```

### Validation Error Reference

| Error Code | Description | Solution |
|------------|-------------|----------|
| `SCHEMA_001` | Missing required field | Add required field to data structure |
| `SCHEMA_002` | Invalid data type | Convert to expected type (int/float/string) |
| `SCHEMA_003` | Value out of range | Check min/max constraints in schema |
| `SCHEMA_004` | Invalid enum value | Use only allowed enum values |
| `BUSINESS_001` | Reading time outside range | Adjust segmentation parameters |
| `BUSINESS_002` | Insufficient focus words | Improve vocabulary extraction algorithm |
| `REFERENCE_001` | Invalid cross-reference | Ensure referenced records exist |

### Performance Optimization

#### For Large-Scale Processing
```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def process_clear_data_parallel(records: List[Dict], max_workers: int = 4):
    """Process CLEAR data using parallel processing."""
    
    # Split records into chunks for parallel processing
    chunk_size = len(records) // max_workers
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_chunk, chunk)
            for chunk in chunks
        ]
        
        results = await asyncio.gather(*tasks)
    
    # Combine results
    all_processed = []
    for chunk_result in results:
        all_processed.extend(chunk_result)
    
    return all_processed
```

This comprehensive documentation provides complete guidance for understanding, validating, and working with all data components in the annotation system. The verbose nature ensures that developers can work independently with the data infrastructure without requiring additional context or documentation.