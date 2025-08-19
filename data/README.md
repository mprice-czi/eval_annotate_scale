# Data Directory

This directory contains sample datasets, schemas, and examples for the annotation system.

## Structure

```
data/
├── schemas/           # JSON schemas for validation
├── examples/          # Sample annotation data
├── fixtures/          # Test fixtures
└── outputs/           # Generated outputs (gitignored)
```

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