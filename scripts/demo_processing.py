#!/usr/bin/env python3
"""
Demo processing script to show the data pipeline structure without API calls.
This demonstrates the processed passage format for dress rehearsal.
"""

import json
import pandas as pd
import sys
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bazel_utils import resolve_workspace_path, ensure_output_directory


@dataclass
class ProcessedPassage:
    """Mock processed passage for demonstration."""
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
    """Mock passage pair for demonstration."""
    pair_id: str
    passage_a: ProcessedPassage
    passage_b: ProcessedPassage
    marginally_decidable: bool
    confidence_score: float
    reasoning: str


def create_demo_processed_passages(max_passages: int = 3) -> List[ProcessedPassage]:
    """Create demo processed passages from CLEAR data."""
    
    # Load real CLEAR data (workspace-relative path)
    clear_csv_path = resolve_workspace_path('data/CLEAR.csv')
    df = pd.read_csv(clear_csv_path, encoding='utf-8-sig')
    df_clean = df[df.iloc[:, 0].notnull()].head(max_passages)
    
    processed_passages = []
    
    for _, row in df_clean.iterrows():
        original_id = str(int(row.iloc[0]))
        excerpt = row.iloc[14]  # Excerpt column
        flesch_score = row.iloc[24]  # Flesch Reading Ease
        
        # Simulate segmentation - split long excerpts into segments
        sentences = excerpt.split('. ')
        segments = []
        current_segment = []
        
        for sentence in sentences:
            current_segment.append(sentence)
            # Create segment when we have 2-3 sentences (simulating 10-15 second reading)
            if len(current_segment) >= 2 and len(' '.join(current_segment)) > 80:
                segments.append('. '.join(current_segment) + '.')
                current_segment = []
        
        # Add remaining sentences
        if current_segment:
            segments.append('. '.join(current_segment) + '.')
        
        # Create processed passages for each segment
        for i, segment_text in enumerate(segments):
            word_count = len(segment_text.split())
            reading_time = min(15.0, word_count / 200 * 60)  # seconds
            
            # Mock complexity estimate based on Flesch score
            if flesch_score >= 70:
                complexity = "Easy"
            elif flesch_score >= 50:
                complexity = "Medium"
            else:
                complexity = "Hard"
            
            # Mock vocabulary focus words (extract longer words)
            words = segment_text.split()
            focus_words = [w.strip('.,!?') for w in words if len(w) > 6 and w.isalpha()][:5]
            
            processed_passage = ProcessedPassage(
                original_id=original_id,
                segment_id=f"{original_id}_seg_{i+1}",
                text=segment_text,
                estimated_reading_time=reading_time,
                flesch_score=flesch_score,
                complexity_estimate=complexity,
                context_preserved=len(segment_text) < 500,
                vocabulary_focus_words=focus_words
            )
            processed_passages.append(processed_passage)
    
    return processed_passages


def create_demo_pairs(processed_passages: List[ProcessedPassage], target_pairs: int = 2) -> List[PassagePair]:
    """Create demo passage pairs (excluding same-original pairs)."""
    
    pairs = []
    pair_count = 0
    
    for i, passage_a in enumerate(processed_passages):
        for j, passage_b in enumerate(processed_passages[i+1:], i+1):
            if pair_count >= target_pairs:
                break
            
            # CRITICAL: Only pair passages from different original sources
            if passage_a.original_id != passage_b.original_id:
                flesch_diff = abs(passage_a.flesch_score - passage_b.flesch_score)
                
                pair = PassagePair(
                    pair_id=f"demo_pair_{pair_count+1:03d}",
                    passage_a=passage_a,
                    passage_b=passage_b,
                    marginally_decidable=5 <= flesch_diff <= 25,
                    confidence_score=0.7 + (pair_count * 0.1),
                    reasoning=f"Demo pair with Flesch difference of {flesch_diff:.1f}. "
                             f"Passage A ({passage_a.complexity_estimate}) vs "
                             f"Passage B ({passage_b.complexity_estimate}) provides "
                             f"appropriate vocabulary complexity comparison."
                )
                pairs.append(pair)
                pair_count += 1
        
        if pair_count >= target_pairs:
            break
    
    return pairs


def main():
    """Run demo processing pipeline."""
    print("🎭 DRESS REHEARSAL: Demo Processing Pipeline")
    print("=" * 50)
    
    # Create output directory (workspace-relative path)
    output_dir = ensure_output_directory("data/outputs")
    
    # Generate demo processed passages
    print("🔄 Generating demo processed passages...")
    processed_passages = create_demo_processed_passages(max_passages=3)
    
    print(f"✅ Generated {len(processed_passages)} processed passages")
    for passage in processed_passages:
        print(f"   - {passage.segment_id}: {passage.complexity_estimate} "
              f"({passage.estimated_reading_time:.1f}s, {len(passage.vocabulary_focus_words)} focus words)")
    
    # Generate demo pairs
    print(f"\\n🔄 Generating demo passage pairs...")
    pairs = create_demo_pairs(processed_passages, target_pairs=3)
    
    print(f"✅ Generated {len(pairs)} passage pairs (different originals only)")
    for pair in pairs:
        print(f"   - {pair.pair_id}: {pair.passage_a.segment_id} + {pair.passage_b.segment_id} "
              f"(confidence: {pair.confidence_score:.2f})")
    
    # Save processed passages
    passages_output = output_dir / "demo_processed_passages.json"
    passages_data = {
        "metadata": {
            "pipeline": "demo_preprocessing",
            "total_passages": len(processed_passages),
            "timestamp": "2025-08-20T11:11:00Z"
        },
        "processed_passages": [asdict(p) for p in processed_passages]
    }
    
    with open(passages_output, 'w') as f:
        json.dump(passages_data, f, indent=2)
    print(f"\\n💾 Saved processed passages to: {passages_output}")
    
    # Save passage pairs
    pairs_output = output_dir / "demo_passage_pairs.json" 
    pairs_data = {
        "metadata": {
            "pipeline": "demo_preprocessing", 
            "total_pairs": len(pairs),
            "timestamp": "2025-08-20T11:11:00Z"
        },
        "passage_pairs": [asdict(p) for p in pairs]
    }
    
    with open(pairs_output, 'w') as f:
        json.dump(pairs_data, f, indent=2)
    print(f"💾 Saved passage pairs to: {pairs_output}")
    
    print(f"\\n🎯 Demo processing complete! Check data/outputs/ for results.")


if __name__ == "__main__":
    main()