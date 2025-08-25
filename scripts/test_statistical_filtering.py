#!/usr/bin/env python3
"""
Test script to validate the statistical pre-filtering approach.

This script creates sample passages with known complexity differences
and tests whether the statistical filter correctly identifies promising pairs.
"""

import sys
from pathlib import Path
from typing import List
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.statistical_pair_filter import StatisticalPairFilter, ProcessedPassage


def create_test_passages() -> List[ProcessedPassage]:
    """Create test passages with varying complexity levels."""
    test_data = [
        # Easy passages
        {
            "original_id": "doc1", "segment_id": "easy-1", "complexity_estimate": "Easy",
            "text": "The cat sat on the mat. It was a sunny day. The birds sang in the trees.",
            "flesch_score": 90.0, "vocabulary_focus_words": ["cat", "mat", "sunny", "birds"]
        },
        {
            "original_id": "doc2", "segment_id": "easy-2", "complexity_estimate": "Easy", 
            "text": "John likes to play ball. He runs fast. The game is fun for him.",
            "flesch_score": 85.0, "vocabulary_focus_words": ["play", "ball", "runs", "game"]
        },
        
        # Medium passages
        {
            "original_id": "doc3", "segment_id": "medium-1", "complexity_estimate": "Medium",
            "text": "The economic implications of technological advancement require careful consideration. Market dynamics shift rapidly in response to innovation cycles.",
            "flesch_score": 45.0, "vocabulary_focus_words": ["economic", "implications", "technological", "advancement"]
        },
        {
            "original_id": "doc4", "segment_id": "medium-2", "complexity_estimate": "Medium",
            "text": "Scientific methodology emphasizes systematic observation and empirical analysis. Research protocols ensure reproducible experimental results.",
            "flesch_score": 40.0, "vocabulary_focus_words": ["methodology", "systematic", "empirical", "protocols"]
        },
        
        # Hard passages  
        {
            "original_id": "doc5", "segment_id": "hard-1", "complexity_estimate": "Hard",
            "text": "The phenomenological approach to epistemological inquiry necessitates a comprehensive understanding of hermeneutic principles and methodological frameworks.",
            "flesch_score": 15.0, "vocabulary_focus_words": ["phenomenological", "epistemological", "hermeneutic", "methodological"]
        },
        {
            "original_id": "doc6", "segment_id": "hard-2", "complexity_estimate": "Hard",
            "text": "Quantum mechanical paradigms demonstrate the fundamental indeterminacy inherent in microscopic physical systems and their macroscopic manifestations.",
            "flesch_score": 10.0, "vocabulary_focus_words": ["quantum", "paradigms", "indeterminacy", "microscopic"]
        }
    ]
    
    passages = []
    for data in test_data:
        passage = ProcessedPassage(
            original_id=data["original_id"],
            segment_id=data["segment_id"],
            text=data["text"],
            estimated_reading_time=len(data["text"].split()) / 200 * 60,  # Rough estimate
            flesch_score=data["flesch_score"],
            complexity_estimate=data["complexity_estimate"],
            context_preserved=True,
            vocabulary_focus_words=data["vocabulary_focus_words"],
            processing_timestamp="2024-01-01T00:00:00",
            source_text_hash="test_hash"
        )
        passages.append(passage)
    
    return passages


def test_statistical_filtering():
    """Test the statistical pre-filtering functionality."""
    print("🧪 Testing Statistical Pre-Filtering")
    print("="*50)
    
    # Create test configuration
    config = {
        "marginality": {
            "confidence_threshold": 0.6,
            "flesch_difference_range": [5, 25],
            "max_candidate_pairs": 100,
            "target_marginal_pairs": 3
        },
        "pairing": {
            "exclude_same_original": True,
            "random_seed": 42
        },
        "statistical_filtering": {
            "enabled": True,
            "max_tfidf_features": 100,  # Smaller for test
            "n_clusters": 3,
            "pca_components": 5,
            "boundary_pairs_ratio": 0.4,
            "marginal_pairs_ratio": 0.3,
            "diverse_pairs_ratio": 0.3
        }
    }
    
    # Create test passages
    passages = create_test_passages()
    print(f"📚 Created {len(passages)} test passages:")
    for p in passages:
        print(f"  - {p.segment_id}: {p.complexity_estimate} (Flesch: {p.flesch_score})")
    print()
    
    # Traditional approach: All possible pairs
    all_pairs = []
    for i, p1 in enumerate(passages):
        for p2 in passages[i+1:]:
            # Skip same original document pairs
            if p1.original_id != p2.original_id:
                flesch_diff = abs(p1.flesch_score - p2.flesch_score)
                if 5 <= flesch_diff <= 25:  # Within marginality range
                    all_pairs.append((p1, p2))
    
    print(f"🔄 Traditional approach: {len(all_pairs)} candidate pairs")
    for p1, p2 in all_pairs:
        flesch_diff = abs(p1.flesch_score - p2.flesch_score)
        print(f"  - {p1.segment_id} vs {p2.segment_id}: Flesch diff = {flesch_diff:.1f}")
    print()
    
    # Statistical filtering approach
    start_time = time.time()
    filter_obj = StatisticalPairFilter(config)
    filtered_pairs = filter_obj.filter_candidate_pairs(passages, max_candidates=10)
    filter_time = time.time() - start_time
    
    print(f"🤖 Statistical filtering: {len(filtered_pairs)} candidate pairs (in {filter_time:.3f}s)")
    for p1, p2 in filtered_pairs:
        flesch_diff = abs(p1.flesch_score - p2.flesch_score)
        print(f"  - {p1.segment_id} vs {p2.segment_id}: Flesch diff = {flesch_diff:.1f}")
    print()
    
    # Analysis
    if filtered_pairs:
        reduction_ratio = len(filtered_pairs) / len(all_pairs) if all_pairs else 0
        print(f"📊 Analysis:")
        print(f"  - Reduction ratio: {reduction_ratio:.2%} of original pairs")
        print(f"  - Processing time: {filter_time:.3f} seconds")
        
        # Check if filtering found good candidates
        good_pairs = 0
        for p1, p2 in filtered_pairs:
            flesch_diff = abs(p1.flesch_score - p2.flesch_score)
            # Good pairs: different complexity levels but within marginal range
            if p1.complexity_estimate != p2.complexity_estimate and 10 <= flesch_diff <= 20:
                good_pairs += 1
        
        print(f"  - Promising pairs found: {good_pairs}/{len(filtered_pairs)} ({good_pairs/len(filtered_pairs):.1%})")
        
        if good_pairs > 0:
            print("✅ Statistical filtering successfully identified promising pairs!")
        else:
            print("⚠️ Statistical filtering may need tuning for better pair selection")
    else:
        print("❌ Statistical filtering found no candidate pairs")
    
    print()
    print("🎯 Expected Behavior:")
    print("  - Should find pairs between different complexity levels")
    print("  - Should exclude same-document pairs")
    print("  - Should prioritize marginal Flesch score differences")
    print("  - Should be much faster than assessing all possible pairs with AI")


if __name__ == "__main__":
    test_statistical_filtering()