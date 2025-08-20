#!/usr/bin/env python3
"""
Intelligent Passage Preprocessing for SuperAnnotate Arm 1a

This script uses Gemini and LangGraph to:
1. Segment CLEAR corpus passages into contextually complete, readable chunks
2. Use agentic workflow to identify marginally decidable passage pairs
3. Generate optimal annotation tasks for vocabulary complexity assessment

Usage:
    python scripts/intelligent_passage_preprocessing.py --output data/marginal_pairs_arm1a.json
"""

import argparse
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessedPassage:
    """Represents a processed passage segment."""
    original_id: str
    segment_id: str
    text: str
    estimated_reading_time: float
    flesch_score: float
    complexity_estimate: str
    context_preserved: bool
    vocabulary_focus_words: List[str]

@dataclass
class PassagePair:
    """Represents a passage pair for annotation."""
    pair_id: str
    passage_a: ProcessedPassage
    passage_b: ProcessedPassage
    marginally_decidable: bool
    confidence_score: float
    reasoning: str

# Pydantic models for structured outputs
class SegmentationResult(BaseModel):
    """Result from passage segmentation."""
    segments: List[Dict] = Field(description="List of passage segments")
    reasoning: str = Field(description="Explanation of segmentation strategy")

class MarginabilityAssessment(BaseModel):
    """Assessment of whether a passage pair is marginally decidable."""
    is_marginal: bool = Field(description="Whether the pair is marginally decidable")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Detailed reasoning for the assessment")
    complexity_difference: str = Field(description="Description of complexity difference")

class PassagePreprocessor:
    """Main class for intelligent passage preprocessing."""
    
    def __init__(self, gemini_api_key: str, config_path: str = None):
        """Initialize with Gemini API key and optional configuration."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize LLM with config values
        gemini_config = self.config.get('gemini', {})
        self.llm = ChatGoogleGenerativeAI(
            model=gemini_config.get('model', 'gemini-1.5-pro'),
            google_api_key=gemini_api_key,
            temperature=gemini_config.get('temperature', 0.3)
        )
        self.processed_passages: List[ProcessedPassage] = []
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Return default configuration if file loading fails
        logger.info("Using default configuration")
        return {
            'gemini': {'model': 'gemini-1.5-pro', 'temperature': 0.3},
            'segmentation': {'target_reading_time_seconds': 12.5},
            'marginality': {'confidence_threshold': 0.6, 'target_marginal_pairs': 50},
            'pairing': {'random_seed': 42}
        }
        
    def load_clear_data(self, csv_path: str) -> pd.DataFrame:
        """Load CLEAR corpus data."""
        logger.info(f"Loading CLEAR data from {csv_path}")
        # Load the CSV - the multiline header is handled properly by pandas
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Excerpt', 'Flesch-Reading-Ease'])
        logger.info(f"Loaded {len(df)} passages with valid excerpts and Flesch scores")
        return df
    
    async def segment_passage(self, original_id: str, text: str, flesch_score: float) -> List[ProcessedPassage]:
        """Segment a passage into contextually complete, readable chunks using Gemini."""
        
        # Get segmentation config
        seg_config = self.config.get('segmentation', {})
        target_time = seg_config.get('target_reading_time_seconds', 12.5)
        word_range = seg_config.get('target_word_count_range', [50, 150])
        min_words, max_words = word_range
        
        segmentation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an expert in educational content analysis. Your task is to segment text passages into smaller chunks that:
1. Can be read in approximately {target_time} seconds (roughly {min_words}-{max_words} words)
2. Maintain complete contextual content for vocabulary complexity assessment
3. Preserve semantic coherence - don't break mid-sentence or mid-thought
4. Each segment should be independently assessable for vocabulary complexity
5. Identify key vocabulary words that indicate complexity in each segment

For each segment, provide:
- The text of the segment
- Estimated word count
- Whether context is preserved (true/false)
- List of 3-5 key vocabulary words that indicate complexity level
- Brief justification for the segmentation points"""),
            
            HumanMessage(content=f"""
Original passage (Flesch Score: {flesch_score}):
{text}

Please segment this passage optimally for vocabulary complexity assessment. Return as JSON with this structure:
{{
    "segments": [
        {{
            "text": "segment text here",
            "word_count": 85,
            "context_preserved": true,
            "vocabulary_focus_words": ["sophisticated", "terminology", "concept"],
            "complexity_rationale": "Contains academic vocabulary and complex concepts"
        }}
    ],
    "reasoning": "Overall segmentation strategy explanation"
}}""")
        ])
        
        parser = JsonOutputParser(pydantic_object=SegmentationResult)
        chain = segmentation_prompt | self.llm | parser
        
        try:
            result = await chain.ainvoke({})
            segments = []
            
            for i, segment_data in enumerate(result["segments"]):
                # Calculate estimated reading time (average 200 words per minute)
                word_count = segment_data["word_count"]
                reading_time = word_count / 200 * 60  # seconds
                
                # Estimate complexity based on Flesch score and vocabulary words
                complexity_estimate = self._estimate_complexity(flesch_score, segment_data["vocabulary_focus_words"])
                
                segment = ProcessedPassage(
                    original_id=original_id,
                    segment_id=f"{original_id}_seg_{i+1}",
                    text=segment_data["text"],
                    estimated_reading_time=reading_time,
                    flesch_score=flesch_score,  # Inherit from original
                    complexity_estimate=complexity_estimate,
                    context_preserved=segment_data["context_preserved"],
                    vocabulary_focus_words=segment_data["vocabulary_focus_words"]
                )
                segments.append(segment)
            
            logger.info(f"Segmented passage {original_id} into {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting passage {original_id}: {e}")
            # Fallback: create single segment if AI fails
            return [ProcessedPassage(
                original_id=original_id,
                segment_id=f"{original_id}_seg_1",
                text=text[:500] + "..." if len(text) > 500 else text,  # Truncate if too long
                estimated_reading_time=min(15.0, len(text.split()) / 200 * 60),
                flesch_score=flesch_score,
                complexity_estimate=self._estimate_complexity(flesch_score, []),
                context_preserved=len(text) <= 500,
                vocabulary_focus_words=[]
            )]
    
    def _estimate_complexity(self, flesch_score: float, vocab_words: List[str]) -> str:
        """Estimate complexity category from Flesch score and vocabulary."""
        if flesch_score >= 70:
            base_complexity = "Easy"
        elif flesch_score >= 50:
            base_complexity = "Medium"
        else:
            base_complexity = "Hard"
        
        # Adjust based on vocabulary sophistication
        sophisticated_indicators = ["sophisticated", "complex", "academic", "technical", "theoretical"]
        if any(word.lower() in " ".join(vocab_words).lower() for word in sophisticated_indicators):
            if base_complexity == "Easy":
                base_complexity = "Medium"
            elif base_complexity == "Medium":
                base_complexity = "Hard"
        
        return base_complexity
    
    async def assess_pair_marginality(self, passage_a: ProcessedPassage, passage_b: ProcessedPassage) -> MarginabilityAssessment:
        """Assess whether a passage pair is marginally decidable for vocabulary complexity."""
        
        marginality_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert annotation task designer. Your job is to identify passage pairs that are "marginally decidable" for vocabulary complexity assessment.

MARGINALLY DECIDABLE means:
- The passages have vocabulary complexity that is close but distinguishable
- An annotator can reasonably determine which is more complex, but it requires careful consideration
- NOT too easy (obviously different complexity levels)
- NOT too hard (essentially identical complexity - would be random guessing)
- The "sweet spot" where human judgment is meaningful and consistent

Consider:
1. Flesch readability scores (closer scores = more marginal)
2. Vocabulary sophistication and academic language use
3. Sentence structure complexity
4. Domain-specific terminology
5. Whether differences are perceivable to general annotators

Rate confidence 0-1 where:
- 0.9+ = Definitely marginal, perfect for annotation
- 0.7-0.9 = Likely marginal, good candidate
- 0.5-0.7 = Possibly marginal, moderate candidate
- Below 0.5 = Not marginal (too easy or too hard)"""),
            
            HumanMessage(content=f"""
Assess this passage pair for marginality in vocabulary complexity:

PASSAGE A (Flesch: {passage_a.flesch_score}):
Vocab focus: {', '.join(passage_a.vocabulary_focus_words)}
Text: {passage_a.text}

PASSAGE B (Flesch: {passage_b.flesch_score}):  
Vocab focus: {', '.join(passage_b.vocabulary_focus_words)}
Text: {passage_b.text}

Return assessment as JSON:
{{
    "is_marginal": true/false,
    "confidence": 0.85,
    "reasoning": "Detailed explanation of why this pair is/isn't marginal",
    "complexity_difference": "Description of the key complexity differences"
}}""")
        ])
        
        parser = JsonOutputParser(pydantic_object=MarginabilityAssessment)
        chain = marginality_prompt | self.llm | parser
        
        try:
            result = await chain.ainvoke({})
            return MarginabilityAssessment(**result)
        except Exception as e:
            logger.error(f"Error assessing marginality: {e}")
            # Fallback assessment based on Flesch score difference
            flesch_diff = abs(passage_a.flesch_score - passage_b.flesch_score)
            is_marginal = 5 <= flesch_diff <= 25  # Reasonable difference range
            return MarginabilityAssessment(
                is_marginal=is_marginal,
                confidence=0.5,
                reasoning=f"Fallback assessment based on Flesch difference: {flesch_diff}",
                complexity_difference=f"Flesch score difference of {flesch_diff} points"
            )

class LangGraphPreprocessingWorkflow:
    """LangGraph workflow for agentic passage pair selection."""
    
    def __init__(self, preprocessor: PassagePreprocessor):
        self.preprocessor = preprocessor
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for preprocessing."""
        
        class WorkflowState(BaseModel):
            original_passages: List[Dict] = Field(default_factory=list)
            segmented_passages: List[ProcessedPassage] = Field(default_factory=list)
            candidate_pairs: List[Tuple[ProcessedPassage, ProcessedPassage]] = Field(default_factory=list)
            marginal_pairs: List[PassagePair] = Field(default_factory=list)
            stage: str = "start"
            batch_size: int = 50  # Process in batches to avoid API limits
        
        workflow = StateGraph(WorkflowState)
        
        async def segment_passages_node(state: WorkflowState) -> WorkflowState:
            """Node: Segment original passages into optimal chunks."""
            logger.info("Segmenting passages...")
            segmented = []
            
            for i, passage_data in enumerate(state.original_passages):
                if i >= state.batch_size:  # Process in batches
                    break
                    
                segments = await self.preprocessor.segment_passage(
                    passage_data["id"],
                    passage_data["text"], 
                    passage_data["flesch_score"]
                )
                segmented.extend(segments)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(state.original_passages)} passages")
            
            state.segmented_passages = segmented
            state.stage = "segmented"
            return state
        
        async def generate_candidate_pairs_node(state: WorkflowState) -> WorkflowState:
            """Node: Generate candidate pairs for marginality assessment."""
            logger.info("Generating candidate pairs...")
            
            # Strategic pairing based on complexity estimates and Flesch scores
            easy_passages = [p for p in state.segmented_passages if p.complexity_estimate == "Easy"]
            medium_passages = [p for p in state.segmented_passages if p.complexity_estimate == "Medium"]
            hard_passages = [p for p in state.segmented_passages if p.complexity_estimate == "Hard"]
            
            candidates = []
            
            # Within-category pairs (should be more marginal)
            # IMPORTANT: Only pair passages from different original sources
            # This ensures annotation tasks are meaningful - subpassages from the same
            # original passage would be too similar and provide poor annotation data
            for category_passages in [easy_passages, medium_passages, hard_passages]:
                if len(category_passages) >= 2:
                    category_pairs = [
                        (p1, p2) for p1, p2 in combinations(category_passages, 2)
                        if p1.original_id != p2.original_id  # Ensure different original passages
                    ]
                    # Sample subset to avoid too many pairs
                    sampled_pairs = random.sample(category_pairs, min(20, len(category_pairs))) if category_pairs else []
                    candidates.extend(sampled_pairs)
            
            # Adjacent-category pairs (Easy-Medium, Medium-Hard)
            # IMPORTANT: Only pair passages from different original sources
            adjacent_pairs = []
            if easy_passages and medium_passages:
                easy_medium = [
                    (e, m) for e in easy_passages[:10] for m in medium_passages[:10]
                    if e.original_id != m.original_id  # Ensure different original passages
                ]
                adjacent_pairs.extend(random.sample(easy_medium, min(15, len(easy_medium))) if easy_medium else [])
            
            if medium_passages and hard_passages:
                medium_hard = [
                    (m, h) for m in medium_passages[:10] for h in hard_passages[:10]
                    if m.original_id != h.original_id  # Ensure different original passages
                ]
                adjacent_pairs.extend(random.sample(medium_hard, min(15, len(medium_hard))) if medium_hard else [])
            
            candidates.extend(adjacent_pairs)
            
            # Shuffle for randomness
            random.shuffle(candidates)
            state.candidate_pairs = candidates[:100]  # Limit for API costs
            state.stage = "candidate_pairs"
            logger.info(f"Generated {len(state.candidate_pairs)} candidate pairs (filtered to exclude same-original pairs)")
            return state
        
        async def assess_marginality_node(state: WorkflowState) -> WorkflowState:
            """Node: Assess marginality of candidate pairs."""
            logger.info("Assessing pair marginality...")
            marginal_pairs = []
            
            for i, (passage_a, passage_b) in enumerate(state.candidate_pairs):
                assessment = await self.preprocessor.assess_pair_marginality(passage_a, passage_b)
                
                confidence_threshold = self.preprocessor.config.get('marginality', {}).get('confidence_threshold', 0.6)
                if assessment.is_marginal and assessment.confidence >= confidence_threshold:
                    pair = PassagePair(
                        pair_id=f"pair_{i+1:04d}",
                        passage_a=passage_a,
                        passage_b=passage_b,
                        marginally_decidable=assessment.is_marginal,
                        confidence_score=assessment.confidence,
                        reasoning=assessment.reasoning
                    )
                    marginal_pairs.append(pair)
                
                if i % 10 == 0:
                    logger.info(f"Assessed {i+1}/{len(state.candidate_pairs)} pairs, found {len(marginal_pairs)} marginal")
            
            # Sort by confidence score
            marginal_pairs.sort(key=lambda x: x.confidence_score, reverse=True)
            state.marginal_pairs = marginal_pairs
            state.stage = "complete"
            logger.info(f"Found {len(marginal_pairs)} marginal pairs")
            return state
        
        # Build workflow graph
        workflow.add_node("segment", segment_passages_node)
        workflow.add_node("generate_pairs", generate_candidate_pairs_node)
        workflow.add_node("assess_marginality", assess_marginality_node)
        
        workflow.set_entry_point("segment")
        workflow.add_edge("segment", "generate_pairs")
        workflow.add_edge("generate_pairs", "assess_marginality")
        workflow.add_edge("assess_marginality", END)
        
        return workflow.compile()
    
    async def run(self, passages_data: List[Dict]) -> List[PassagePair]:
        """Run the complete preprocessing workflow."""
        initial_state = {
            "original_passages": passages_data,
            "segmented_passages": [],
            "candidate_pairs": [],
            "marginal_pairs": [],
            "stage": "start"
        }
        
        result = await self.workflow.ainvoke(initial_state)
        return result["marginal_pairs"]

async def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Intelligent passage preprocessing for SuperAnnotate')
    parser.add_argument('--clear-csv', default='data/CLEAR.csv', help='Path to CLEAR.csv')
    parser.add_argument('--output', default='data/outputs/marginal_pairs_arm1a.json', help='Output JSON file')
    parser.add_argument('--config', default='configs/preprocessing_config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--max-passages', type=int, default=100, help='Maximum passages to process')
    parser.add_argument('--target-pairs', type=int, default=50, help='Target number of marginal pairs')
    
    args = parser.parse_args()
    
    # Secure API key retrieval from environment
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        # Check alternative environment variable names
        gemini_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_AI_API_KEY')
        
    if not gemini_api_key:
        logger.error("❌ Gemini API key not found!")
        logger.error("Please set one of these environment variables:")
        logger.error("  export GEMINI_API_KEY='your-api-key'")
        logger.error("  export GOOGLE_API_KEY='your-api-key'") 
        logger.error("  export GOOGLE_AI_API_KEY='your-api-key'")
        logger.error("\nNever put API keys directly in code or command line arguments!")
        return
    
    # Initialize preprocessor with secure API key and configuration
    preprocessor = PassagePreprocessor(gemini_api_key, args.config)
    
    # Load CLEAR data
    df = preprocessor.load_clear_data(args.clear_csv)
    
    # Sample passages for processing (to manage API costs)
    if len(df) > args.max_passages:
        random_seed = preprocessor.config.get('pairing', {}).get('random_seed', 42)
        df = df.sample(args.max_passages, random_state=random_seed)
    
    # Prepare passage data
    passages_data = []
    for _, row in df.iterrows():
        passages_data.append({
            "id": str(row["ID"]),
            "text": row["Excerpt"],
            "flesch_score": float(row["Flesch-Reading-Ease"])
        })
    
    # Create and run workflow
    workflow = LangGraphPreprocessingWorkflow(preprocessor)
    marginal_pairs = await workflow.run(passages_data)
    
    # Export results
    target_pairs = preprocessor.config.get('marginality', {}).get('target_marginal_pairs', args.target_pairs)
    output_data = {
        "metadata": {
            "total_original_passages": len(passages_data),
            "total_marginal_pairs": len(marginal_pairs),
            "target_pairs": target_pairs,
            "selection_criteria": "Marginal decidability for vocabulary complexity assessment",
            "config_used": preprocessor.config
        },
        "pairs": [asdict(pair) for pair in marginal_pairs[:target_pairs]]
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(marginal_pairs)} marginal pairs to {output_path}")
    
    # Summary statistics
    avg_confidence = sum(pair.confidence_score for pair in marginal_pairs) / len(marginal_pairs)
    logger.info(f"Average confidence score: {avg_confidence:.3f}")
    
    complexity_distribution = {}
    for pair in marginal_pairs:
        key = f"{pair.passage_a.complexity_estimate}-{pair.passage_b.complexity_estimate}"
        complexity_distribution[key] = complexity_distribution.get(key, 0) + 1
    
    logger.info(f"Complexity distribution: {complexity_distribution}")

if __name__ == "__main__":
    asyncio.run(main())