#!/usr/bin/env python3
"""
Passage Segmentation Script - Stage 1 of Intelligent Processing Pipeline

This script segments CLEAR corpus passages into contextually complete, readable chunks
using Gemini AI with robust failure handling and caching capabilities.

Features:
- Resumable processing with progress tracking
- Intermediate result caching
- Batch processing with configurable limits
- Detailed error reporting and recovery
- JSON Schema validation

Usage:
    bazel run //scripts:segment_passages -- \
        --config configs/preprocessing_config.yaml \
        --output data/outputs/segmented_passages.json \
        --max-passages 100 \
        --resume
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
import hashlib
import time

import pandas as pd
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_manager import SecureConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segment_passages.log'),
        logging.StreamHandler()
    ]
)
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
    processing_timestamp: str
    source_text_hash: str  # For cache invalidation

class SegmentationResult(BaseModel):
    """Result from passage segmentation."""
    segments: List[Dict] = Field(description="List of passage segments")
    reasoning: str = Field(description="Explanation of segmentation strategy")

class PassageSegmenter:
    """Robust passage segmentation with caching and recovery."""
    
    def __init__(self, config: Dict, api_key: str):
        """Initialize segmenter with configuration."""
        self.config = config
        self.gemini_config = config.get('gemini', {})
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.gemini_config.get('model', 'gemini-2.5-pro'),
            google_api_key=api_key,
            temperature=self.gemini_config.get('temperature', 0.3),
            timeout=self.gemini_config.get('timeout_seconds', 30),
            max_retries=self.gemini_config.get('max_retries', 3)
        )
        
        # Processing state
        self.processed_passages: List[ProcessedPassage] = []
        self.processed_ids: Set[str] = set()
        self.cache_file: Optional[Path] = None
        self.progress_file: Optional[Path] = None
        
        # Rate limiting
        self.last_api_call = 0
        self.min_delay = self.config.get('limits', {}).get('delay_between_requests_ms', 100) / 1000
        
    def setup_caching(self, output_path: Path) -> None:
        """Set up caching and progress tracking files."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache and progress files
        cache_name = f"{output_path.stem}_cache.json"
        progress_name = f"{output_path.stem}_progress.json"
        
        self.cache_file = output_dir / cache_name
        self.progress_file = output_dir / progress_name
        
        logger.info(f"Cache file: {self.cache_file}")
        logger.info(f"Progress file: {self.progress_file}")
    
    def load_progress(self) -> Dict:
        """Load processing progress from file."""
        if not self.progress_file or not self.progress_file.exists():
            return {
                "completed_ids": [],
                "failed_ids": [],
                "last_processed_index": -1,
                "total_processed": 0,
                "start_time": time.time()
            }
        
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                logger.info(f"Resuming from progress: {progress['total_processed']} passages processed")
                return progress
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
            return {
                "completed_ids": [],
                "failed_ids": [],
                "last_processed_index": -1,
                "total_processed": 0,
                "start_time": time.time()
            }
    
    def save_progress(self, progress: Dict) -> None:
        """Save processing progress to file."""
        if not self.progress_file:
            return
            
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_cached_results(self) -> List[ProcessedPassage]:
        """Load cached segmentation results."""
        if not self.cache_file or not self.cache_file.exists():
            return []
        
        try:
            with open(self.cache_file, 'r') as f:
                cached_data = json.load(f)
                
            passages = []
            for passage_data in cached_data:
                passage = ProcessedPassage(**passage_data)
                passages.append(passage)
                self.processed_ids.add(passage.original_id)
            
            logger.info(f"Loaded {len(passages)} cached passages")
            return passages
            
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
            return []
    
    def save_cached_results(self, passages: List[ProcessedPassage]) -> None:
        """Save segmentation results to cache."""
        if not self.cache_file:
            return
            
        try:
            cache_data = [asdict(passage) for passage in passages]
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Cached {len(passages)} passages")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _create_text_hash(self, text: str) -> str:
        """Create hash of text for cache invalidation."""
        return hashlib.md5(text.encode()).hexdigest()[:8]
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)
        self.last_api_call = time.time()
    
    async def segment_passage(self, original_id: str, text: str, flesch_score: float) -> List[ProcessedPassage]:
        """Segment a passage into readable chunks using Gemini."""
        
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
- The text of the segment (complete sentences only)
- Estimated word count
- Whether context is preserved (true/false)
- List of 3-5 key vocabulary words that indicate complexity level
- Brief justification for the segmentation points

IMPORTANT: Return valid JSON only, no additional text."""),
            
            HumanMessage(content=f"""Original passage (Flesch Score: {flesch_score}):
{text}

Please segment this passage optimally for vocabulary complexity assessment. Return as JSON:

{{
    "segments": [
        {{
            "text": "Complete segment text here",
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
        
        # Apply rate limiting
        await self._rate_limit()
        
        try:
            logger.debug(f"Segmenting passage {original_id}")
            result = await chain.ainvoke({})
            segments = []
            
            text_hash = self._create_text_hash(text)
            timestamp = pd.Timestamp.now().isoformat()
            
            for i, segment_data in enumerate(result["segments"]):
                # Calculate estimated reading time (average 200 words per minute)
                word_count = segment_data.get("word_count", len(segment_data["text"].split()))
                reading_time = word_count / 200 * 60  # seconds
                
                # Estimate complexity based on Flesch score and vocabulary words
                complexity_estimate = self._estimate_complexity(flesch_score, segment_data.get("vocabulary_focus_words", []))
                
                segment = ProcessedPassage(
                    original_id=original_id,
                    segment_id=f"{original_id}-seg-{i+1}",
                    text=segment_data["text"],
                    estimated_reading_time=reading_time,
                    flesch_score=flesch_score,
                    complexity_estimate=complexity_estimate,
                    context_preserved=segment_data.get("context_preserved", True),
                    vocabulary_focus_words=segment_data.get("vocabulary_focus_words", []),
                    processing_timestamp=timestamp,
                    source_text_hash=text_hash
                )
                segments.append(segment)
            
            logger.info(f"✅ Segmented passage {original_id} into {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Error segmenting passage {original_id}: {e}")
            # Create fallback segment to avoid complete failure
            return [self._create_fallback_segment(original_id, text, flesch_score)]
    
    def _create_fallback_segment(self, original_id: str, text: str, flesch_score: float) -> ProcessedPassage:
        """Create a fallback segment when AI processing fails."""
        words = text.split()
        truncated_text = ' '.join(words[:150]) + ('...' if len(words) > 150 else '')
        
        return ProcessedPassage(
            original_id=original_id,
            segment_id=f"{original_id}-seg-1",
            text=truncated_text,
            estimated_reading_time=min(15.0, len(words[:150]) / 200 * 60),
            flesch_score=flesch_score,
            complexity_estimate=self._estimate_complexity_from_flesch(flesch_score),
            context_preserved=False,  # Mark as compromised
            vocabulary_focus_words=[],
            processing_timestamp=pd.Timestamp.now().isoformat(),
            source_text_hash=self._create_text_hash(text)
        )
    
    def _estimate_complexity(self, flesch_score: float, vocab_words: List[str]) -> str:
        """Estimate complexity from Flesch score and vocabulary words."""
        # Base complexity from Flesch score
        if flesch_score >= 80:
            base_complexity = "Easy"
        elif flesch_score >= 60:
            base_complexity = "Medium"  
        elif flesch_score >= 40:
            base_complexity = "Hard"
        else:
            base_complexity = "Very Hard"
        
        # Adjust for vocabulary complexity
        advanced_words = len([word for word in vocab_words 
                             if len(word) > 8 or any(suffix in word.lower() 
                                                   for suffix in ['tion', 'ment', 'ness', 'ity'])])
        
        if advanced_words >= 3 and base_complexity == "Easy":
            return "Medium"
        elif advanced_words >= 4 and base_complexity == "Medium":
            return "Hard"
        
        return base_complexity
    
    def _estimate_complexity_from_flesch(self, flesch_score: float) -> str:
        """Simple complexity estimate from Flesch score only."""
        if flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 60:
            return "Medium"
        elif flesch_score >= 40:
            return "Hard"
        else:
            return "Very Hard"
    
    async def process_passages(self, df: pd.DataFrame, max_passages: int = None, resume: bool = False) -> List[ProcessedPassage]:
        """Process multiple passages with progress tracking and error recovery."""
        
        # Load existing progress and cache
        if resume:
            progress = self.load_progress()
            self.processed_passages = self.load_cached_results()
            completed_ids = set(progress["completed_ids"])
            failed_ids = set(progress["failed_ids"])
        else:
            progress = {
                "completed_ids": [],
                "failed_ids": [],
                "last_processed_index": -1,
                "total_processed": 0,
                "start_time": time.time()
            }
            completed_ids = set()
            failed_ids = set()
        
        # Filter passages to process
        if max_passages:
            df = df.head(max_passages)
        
        total_passages = len(df)
        logger.info(f"Processing {total_passages} passages (resume={resume})")
        
        # Process passages in batches
        batch_size = self.config.get('limits', {}).get('max_passages_per_batch', 10)
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_results = []
            
            for idx, row in batch_df.iterrows():
                original_id = str(int(float(row['ID'])))  # Convert float to int to string to remove .0
                
                # Skip if already processed successfully
                if original_id in completed_ids:
                    logger.debug(f"Skipping already processed passage {original_id}")
                    continue
                
                # Skip if previously failed (but could retry with flag)
                if original_id in failed_ids:
                    logger.debug(f"Skipping previously failed passage {original_id}")
                    continue
                
                try:
                    text = str(row['Excerpt'])
                    flesch_score = float(row['Flesch-Reading-Ease'])
                    
                    # Process the passage
                    segments = await self.segment_passage(original_id, text, flesch_score)
                    self.processed_passages.extend(segments)
                    batch_results.extend(segments)
                    
                    # Update progress
                    progress["completed_ids"].append(original_id)
                    progress["total_processed"] += 1
                    progress["last_processed_index"] = idx
                    
                    # Save progress periodically
                    if progress["total_processed"] % 5 == 0:
                        self.save_progress(progress)
                        self.save_cached_results(self.processed_passages)
                    
                    logger.info(f"Progress: {progress['total_processed']}/{total_passages}")
                    
                except Exception as e:
                    logger.error(f"Failed to process passage {original_id}: {e}")
                    progress["failed_ids"].append(original_id)
            
            # Save batch results
            if batch_results:
                self.save_cached_results(self.processed_passages)
                logger.info(f"Completed batch: {len(batch_results)} new segments")
            
            # Rate limiting between batches
            if i + batch_size < len(df):
                await asyncio.sleep(1)
        
        # Final save
        self.save_progress(progress)
        self.save_cached_results(self.processed_passages)
        
        logger.info(f"✅ Processing complete: {len(self.processed_passages)} total segments")
        return self.processed_passages

def load_clear_data(csv_path: str) -> pd.DataFrame:
    """Load and validate CLEAR corpus data."""
    logger.info(f"Loading CLEAR data from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df = df.dropna(subset=['Excerpt', 'Flesch-Reading-Ease'])
        
        # Validate required columns
        required_cols = ['ID', 'Excerpt', 'Flesch-Reading-Ease']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} valid passages")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load CLEAR data: {e}")
        raise

def save_segmented_passages(passages: List[ProcessedPassage], output_path: Path) -> None:
    """Save segmented passages to JSON file with validation."""
    try:
        # Convert to serializable format
        output_data = {
            "metadata": {
                "total_passages": len(passages),
                "generation_timestamp": pd.Timestamp.now().isoformat(),
                "schema_version": "1.0"
            },
            "segmented_passages": [asdict(passage) for passage in passages]
        }
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✅ Saved {len(passages)} segmented passages to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save segmented passages: {e}")
        raise

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Segment CLEAR passages using Gemini AI")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--output", required=True, help="Output path for segmented passages JSON")
    parser.add_argument("--max-passages", type=int, help="Maximum number of passages to process")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--clear-csv", default="data/CLEAR.csv", help="Path to CLEAR.csv file")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get API key
    config_manager = SecureConfigManager()
    api_key = config_manager.get_gemini_api_key()
    if not api_key:
        logger.error("Gemini API key not found. Run setup_bazel_env.py first.")
        return 1
    
    # Load CLEAR data
    try:
        df = load_clear_data(args.clear_csv)
    except Exception as e:
        logger.error(f"Failed to load CLEAR data: {e}")
        return 1
    
    # Initialize segmenter
    segmenter = PassageSegmenter(config, api_key)
    segmenter.setup_caching(Path(args.output))
    
    try:
        # Process passages
        passages = await segmenter.process_passages(
            df, 
            max_passages=args.max_passages, 
            resume=args.resume
        )
        
        # Save results
        save_segmented_passages(passages, Path(args.output))
        
        # Clean up cache and progress files on successful completion
        if not args.resume:  # Keep files if resuming for debugging
            if segmenter.cache_file and segmenter.cache_file.exists():
                segmenter.cache_file.unlink()
            if segmenter.progress_file and segmenter.progress_file.exists():
                segmenter.progress_file.unlink()
        
        logger.info("🎉 Segmentation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))