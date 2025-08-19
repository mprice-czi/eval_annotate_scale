# Prompts Directory

This directory contains LangGraph prompts and templates for annotation workflows.

## Structure

```
prompts/
├── classification/    # Classification task prompts
├── extraction/        # Information extraction prompts
├── evaluation/        # Evaluation and scoring prompts
├── generation/        # Content generation prompts
└── templates/         # Reusable prompt templates
```

## Prompt Categories

### Classification Prompts
- Content type classification
- Quality assessment prompts  
- Educational level classification
- Subject matter categorization

### Extraction Prompts
- Key concept extraction
- Learning objective identification
- Skill and competency extraction
- Metadata extraction

### Evaluation Prompts
- Rubric-based evaluation
- Automated scoring prompts
- Quality assessment criteria
- Comparative evaluation

### Generation Prompts
- Feedback generation
- Summary creation
- Question generation
- Content enhancement

## Usage in LangGraph

Prompts are integrated into LangGraph workflows as:
- Node instructions for workflow steps
- State transformation templates
- Dynamic prompt generation
- Context-aware responses

## Best Practices

- Use Jinja2 templating for dynamic content
- Include clear instructions and examples
- Specify expected output formats
- Test prompts with various input types
- Version control prompt changes