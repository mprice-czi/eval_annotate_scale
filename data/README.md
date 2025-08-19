# Data Directory

This directory contains datasets, schemas, and examples for the annotation system, including the CLEAR corpus.

## Structure

```
data/
├── CLEAR.csv          # CLEAR corpus dataset (6.1MB, 13,203 texts)
├── schemas/           # JSON schemas for validation
├── examples/          # Sample annotation data
├── fixtures/          # Test fixtures
└── outputs/           # Generated outputs (gitignored)
```

## CLEAR Corpus

The **CLEAR** (Corpus of Linguistic and Educational Analysis Resources) dataset is the primary corpus for this annotation project. 

### Dataset Overview
- **File**: `CLEAR.csv` (6.1MB, extracted from CLEAR.csv.zip)
- **Records**: 13,203 text samples with readability and complexity metrics
- **Columns**: 17 fields including text metadata, excerpts, and linguistic features
- **Time Period**: Publication years ranging from early 20th century to modern
- **Sources**: Diverse including Project Gutenberg literature and other collections

### Column Schema
The CLEAR.csv contains the following key columns:
1. **ID** - Unique identifier
2. **Author** - Text author
3. **Title** - Work title  
4. **Source** - Publication source (e.g., gutenberg)
5. **Pub Year** - Publication year
6. **Category** - Text category (e.g., Lit for Literature)
7. **Location** - Geographic/cultural context
8. **License** - Usage rights (PD, PG, etc.)
9. **MPAA ratings** - Content rating metrics
10. **Excerpt** - Text sample for analysis
11. **Readability Metrics** - Flesch-Reading-Ease, Flesch-Kincaid-Grade-Level, etc.
12. **Complexity Measures** - SMOG, Dale-Chall, CAREC, CARES, CML2RI
13. **Predictions** - ML model predictions (firstPlace_pred through sixthPlace_pred)
14. **Kaggle split** - Train/test designation

### Usage Notes
- Headers span multiple lines (lines 1-8) due to complex column names
- Actual data begins at line 9
- Text excerpts contain educational content suitable for annotation tasks
- Readability scores provide baseline complexity measurements

## Dataset Types

### Educational Content
- Student submissions and responses
- Curriculum materials and assessments
- Learning objective mappings

### Annotation Schemas
- Classification task schemas
- Extraction task schemas  
- Evaluation rubrics and criteria

### Sample Data
- Anonymized and de-identified examples
- Synthetic data for testing
- Schema validation examples

## Usage

All datasets in this directory are for development and testing purposes only. Real production data should never be committed to version control.

- Use `schemas/` for JSON Schema definitions
- Place sample data in `examples/` subdirectories
- Test fixtures go in `fixtures/` for unit tests