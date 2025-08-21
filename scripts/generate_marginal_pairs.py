#!/usr/bin/env python3
"""
Marginal Pair Generation Script - Stage 2 of Intelligent Processing Pipeline

This script generates marginally decidable passage pairs from segmented passages
using intelligent filtering and AI assessment for vocabulary complexity annotation.

Features:
- Reads segmented passages from Stage 1 output
- Filters candidates based on business rules
- AI-powered marginality assessment
- Quality scoring and ranking
- Configurable pair selection criteria

Usage:
    bazel run //scripts:generate_marginal_pairs -- \
        --input data/outputs/segmented_passages.json \
        --config configs/preprocessing_config.yaml \
        --output data/outputs/marginal_pairs.json \
        --target-pairs 50
"""

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bazel_utils import resolve_output_file, resolve_workspace_path
from src.config_manager import SecureConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("generate_marginal_pairs.log"),
        logging.StreamHandler(),
    ],
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
    source_text_hash: str


@dataclass
class MarginalPair:
    """Represents a marginally decidable passage pair."""

    pair_id: str
    passage_a: ProcessedPassage
    passage_b: ProcessedPassage
    marginality_confidence: float
    complexity_difference: float
    reasoning: str
    target_annotation_time: float
    pair_quality_score: float
    generation_timestamp: str


class MarginabilityAssessment(BaseModel):
    """Assessment of whether a passage pair is marginally decidable."""

    is_marginal: bool = Field(description="Whether the pair is marginally decidable")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Detailed reasoning for the assessment")
    complexity_difference: str = Field(
        description="Description of complexity difference"
    )
    estimated_annotation_time: float = Field(
        description="Estimated human annotation time in seconds"
    )


class MarginalPairGenerator:
    """Generates marginally decidable passage pairs with quality assessment."""

    def __init__(self, config: Dict, api_key: str):
        """Initialize generator with configuration."""
        self.config = config
        self.gemini_config = config.get("gemini", {})
        self.marginality_config = config.get("marginality", {})
        self.pairing_config = config.get("pairing", {})
        self.quality_config = config.get("quality", {})

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.gemini_config.get("model", "gemini-2.5-pro"),
            google_api_key=api_key,
            temperature=self.gemini_config.get("temperature", 0.3),
            timeout=self.gemini_config.get("timeout_seconds", 30),
            max_retries=self.gemini_config.get("max_retries", 3),
        )

        # Rate limiting
        self.last_api_call = 0
        self.min_delay = (
            self.config.get("limits", {}).get("delay_between_requests_ms", 100) / 1000
        )

        # Set random seed for reproducible sampling
        random.seed(self.pairing_config.get("random_seed", 42))

        # Processing state for caching
        self.processed_pairs: List[MarginalPair] = []
        self.processed_pair_hashes: Set[str] = set()
        self.cache_file: Optional[Path] = None
        self.progress_file: Optional[Path] = None

    def setup_caching(self, output_path: Path) -> None:
        """Set up caching and progress tracking files."""
        from src.bazel_utils import resolve_output_file

        # Resolve output path relative to workspace root
        resolved_output_path = resolve_output_file(output_path)
        output_dir = resolved_output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cache and progress files
        cache_name = f"{resolved_output_path.stem}_cache.json"
        progress_name = f"{resolved_output_path.stem}_progress.json"

        self.cache_file = output_dir / cache_name
        self.progress_file = output_dir / progress_name

        logger.info(f"Cache file: {self.cache_file}")
        logger.info(f"Progress file: {self.progress_file}")

    def load_progress(self) -> Dict:
        """Load processing progress from file."""
        if not self.progress_file or not self.progress_file.exists():
            return {
                "completed_pair_hashes": [],
                "failed_pair_hashes": [],
                "total_processed": 0,
                "start_time": time.time(),
            }

        try:
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
                logger.info(
                    f"Resuming from progress: {progress['total_processed']} pairs processed"
                )
                return progress
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
            return {
                "completed_pair_hashes": [],
                "failed_pair_hashes": [],
                "total_processed": 0,
                "start_time": time.time(),
            }

    def save_progress(self, progress: Dict) -> None:
        """Save processing progress to file."""
        if not self.progress_file:
            return

        try:
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def load_cached_results(self) -> List[MarginalPair]:
        """Load cached marginal pair results."""
        if not self.cache_file or not self.cache_file.exists():
            return []

        try:
            with open(self.cache_file, "r") as f:
                cached_data = json.load(f)

            pairs = []
            for pair_data in cached_data:
                # Reconstruct ProcessedPassage objects
                passage_a_data = pair_data["passage_a"]
                passage_b_data = pair_data["passage_b"]

                passage_a = ProcessedPassage(**passage_a_data)
                passage_b = ProcessedPassage(**passage_b_data)

                pair = MarginalPair(
                    pair_id=pair_data["pair_id"],
                    passage_a=passage_a,
                    passage_b=passage_b,
                    marginality_confidence=pair_data["marginality_confidence"],
                    complexity_difference=pair_data["complexity_difference"],
                    reasoning=pair_data["reasoning"],
                    target_annotation_time=pair_data["target_annotation_time"],
                    pair_quality_score=pair_data["pair_quality_score"],
                    generation_timestamp=pair_data["generation_timestamp"],
                )
                pairs.append(pair)
                self.processed_pair_hashes.add(
                    self._create_pair_hash(passage_a, passage_b)
                )

            logger.info(f"Loaded {len(pairs)} cached pairs")
            return pairs

        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
            return []

    def save_cached_results(self, pairs: List[MarginalPair]) -> None:
        """Save marginal pair results to cache."""
        if not self.cache_file:
            return

        try:
            cache_data = []
            for pair in pairs:
                pair_dict = asdict(pair)
                cache_data.append(pair_dict)

            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Cached {len(pairs)} pairs")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _create_pair_hash(
        self, passage_a: ProcessedPassage, passage_b: ProcessedPassage
    ) -> str:
        """Create hash for pair to check if already processed."""
        # Sort by segment_id for consistent hashing regardless of order
        ids = sorted([passage_a.segment_id, passage_b.segment_id])
        pair_key = f"{ids[0]}#{ids[1]}"
        return hashlib.md5(pair_key.encode()).hexdigest()[:8]

    async def _rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)
        self.last_api_call = time.time()

    def _filter_candidate_pairs(
        self, passages: List[ProcessedPassage]
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Filter passage pairs based on business rules and strategic pairing."""

        logger.info(f"Filtering candidate pairs from {len(passages)} passages")

        # Apply quality filtering first
        filtered_passages = self._apply_quality_filters(passages)
        logger.info(f"After quality filtering: {len(filtered_passages)} passages")

        # Group passages by complexity for strategic pairing
        complexity_groups = self._group_by_complexity(filtered_passages)

        # Generate candidates based on pairing strategy
        candidates = self._generate_strategic_pairs(complexity_groups)

        # Limit candidates for API cost management
        max_candidates = self.marginality_config.get("max_candidate_pairs", 100)
        if len(candidates) > max_candidates:
            # Prioritize pairs with more diverse complexity differences
            candidates.sort(
                key=lambda x: abs(x[0].flesch_score - x[1].flesch_score), reverse=True
            )
            candidates = candidates[:max_candidates]
            logger.info(
                f"Limited to top {max_candidates} candidates by Flesch score diversity"
            )

        return candidates

    def _apply_quality_filters(
        self, passages: List[ProcessedPassage]
    ) -> List[ProcessedPassage]:
        """Apply quality control filters to passages."""
        require_context = self.quality_config.get("require_context_preservation", True)
        min_vocab_words = self.quality_config.get("min_vocabulary_focus_words", 2)
        exclude_errors = self.quality_config.get("exclude_segments_with_errors", True)

        filtered = []
        for passage in passages:
            # Filter by context preservation
            if require_context and not passage.context_preserved:
                if exclude_errors:
                    continue

            # Filter by vocabulary focus words
            if len(passage.vocabulary_focus_words) < min_vocab_words:
                if exclude_errors:
                    continue

            filtered.append(passage)

        return filtered

    def _group_by_complexity(
        self, passages: List[ProcessedPassage]
    ) -> Dict[str, List[ProcessedPassage]]:
        """Group passages by complexity estimate."""
        groups = {"Easy": [], "Medium": [], "Hard": [], "Very Hard": []}
        for passage in passages:
            groups[passage.complexity_estimate].append(passage)
        return groups

    def _generate_strategic_pairs(
        self, complexity_groups: Dict[str, List[ProcessedPassage]]
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Generate pairs based on configured pairing strategy."""
        candidates = []

        # Get pairing configuration
        within_category_max = self.pairing_config.get("within_category_pairs", 20)
        adjacent_category_max = self.pairing_config.get("adjacent_category_pairs", 15)
        cross_category_skip = self.pairing_config.get("cross_category_skip", True)
        exclude_same_original = self.pairing_config.get("exclude_same_original", True)

        complexity_levels = ["Easy", "Medium", "Hard", "Very Hard"]

        # Within-category pairs
        for level in complexity_levels:
            pairs_in_category = self._generate_within_category_pairs(
                complexity_groups[level], within_category_max, exclude_same_original
            )
            candidates.extend(pairs_in_category)
            logger.debug(
                f"Generated {len(pairs_in_category)} within-category pairs for {level}"
            )

        # Adjacent category pairs
        for i in range(len(complexity_levels) - 1):
            level_a = complexity_levels[i]
            level_b = complexity_levels[i + 1]
            adjacent_pairs = self._generate_cross_category_pairs(
                complexity_groups[level_a],
                complexity_groups[level_b],
                adjacent_category_max,
                exclude_same_original,
            )
            candidates.extend(adjacent_pairs)
            logger.debug(
                f"Generated {len(adjacent_pairs)} adjacent pairs: {level_a} vs {level_b}"
            )

        # Skip cross-category pairs if configured (e.g., Easy vs Hard)
        if not cross_category_skip:
            # Add Easy-Hard and Medium-Very Hard pairs
            for level_a, level_b in [("Easy", "Hard"), ("Medium", "Very Hard")]:
                cross_pairs = self._generate_cross_category_pairs(
                    complexity_groups[level_a],
                    complexity_groups[level_b],
                    10,
                    exclude_same_original,  # Fewer cross-category pairs
                )
                candidates.extend(cross_pairs)
                logger.debug(
                    f"Generated {len(cross_pairs)} cross-category pairs: {level_a} vs {level_b}"
                )

        logger.info(
            f"Generated {len(candidates)} total candidate pairs using strategic pairing"
        )
        return candidates

    def _generate_within_category_pairs(
        self,
        passages: List[ProcessedPassage],
        max_pairs: int,
        exclude_same_original: bool,
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Generate pairs within the same complexity category."""
        if len(passages) < 2:
            return []

        candidates = []
        flesch_range = self.marginality_config.get("flesch_difference_range", [5, 25])
        min_diff, max_diff = flesch_range

        # Group by original document if excluding same-original pairs
        if exclude_same_original:
            doc_groups = {}
            for passage in passages:
                doc_groups.setdefault(passage.original_id, []).append(passage)

            # Generate pairs between different documents
            doc_ids = list(doc_groups.keys())
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    for passage_a in doc_groups[doc_ids[i]]:
                        for passage_b in doc_groups[doc_ids[j]]:
                            if self._passes_basic_filters(
                                passage_a, passage_b, min_diff, max_diff
                            ):
                                candidates.append((passage_a, passage_b))
                                if len(candidates) >= max_pairs:
                                    return candidates
        else:
            # Generate pairs from all passages
            for passage_a, passage_b in combinations(passages, 2):
                if self._passes_basic_filters(passage_a, passage_b, min_diff, max_diff):
                    candidates.append((passage_a, passage_b))
                    if len(candidates) >= max_pairs:
                        return candidates

        return candidates

    def _generate_cross_category_pairs(
        self,
        passages_a: List[ProcessedPassage],
        passages_b: List[ProcessedPassage],
        max_pairs: int,
        exclude_same_original: bool,
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Generate pairs between different complexity categories."""
        if not passages_a or not passages_b:
            return []

        candidates = []
        flesch_range = self.marginality_config.get("flesch_difference_range", [5, 25])
        min_diff, max_diff = flesch_range

        for passage_a in passages_a:
            for passage_b in passages_b:
                # Skip if from same original document and configured to exclude
                if (
                    exclude_same_original
                    and passage_a.original_id == passage_b.original_id
                ):
                    continue

                if self._passes_basic_filters(passage_a, passage_b, min_diff, max_diff):
                    candidates.append((passage_a, passage_b))
                    if len(candidates) >= max_pairs:
                        return candidates

        return candidates

    def _passes_basic_filters(
        self,
        passage_a: ProcessedPassage,
        passage_b: ProcessedPassage,
        min_diff: float,
        max_diff: float,
    ) -> bool:
        """Check if a pair passes basic filtering criteria."""
        # Filter by Flesch score difference
        flesch_diff = abs(passage_a.flesch_score - passage_b.flesch_score)
        if not (min_diff <= flesch_diff <= max_diff):
            return False

        # Filter by similar reading times
        time_diff = abs(
            passage_a.estimated_reading_time - passage_b.estimated_reading_time
        )
        if time_diff > 10:  # Within 10 seconds
            return False

        return True

    async def _assess_marginality(
        self, passage_a: ProcessedPassage, passage_b: ProcessedPassage
    ) -> Optional[MarginabilityAssessment]:
        """Use AI to assess if a passage pair is marginally decidable."""

        marginality_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an expert in educational assessment and vocabulary complexity evaluation. Your task is to assess whether two passage pairs are "marginally decidable" for vocabulary complexity comparison.

A marginally decidable pair means:
1. There is a clear but not obvious difference in vocabulary complexity
2. The difference is significant enough to have a definitive answer
3. The difference is subtle enough to require careful consideration (not immediately obvious)
4. Human annotators can reasonably determine which passage is more complex in 30-60 seconds
5. The complexity difference is primarily in vocabulary, not just sentence structure or content

Avoid pairs that are:
- Too similar (no clear difference)
- Too different (answer is obvious immediately)
- Different primarily in topic complexity rather than vocabulary complexity
- Different primarily in sentence structure rather than word choice

Return your assessment as JSON with confidence score (0.0-1.0) and detailed reasoning."""
                ),
                HumanMessage(
                    content=f"""
PASSAGE A (Flesch: {passage_a.flesch_score}, Complexity: {passage_a.complexity_estimate}):
"{passage_a.text}"
Focus words: {passage_a.vocabulary_focus_words}

PASSAGE B (Flesch: {passage_b.flesch_score}, Complexity: {passage_b.complexity_estimate}):
"{passage_b.text}"
Focus words: {passage_b.vocabulary_focus_words}

Assess whether this pair is marginally decidable for vocabulary complexity comparison. Return JSON:

{{
    "is_marginal": true/false,
    "confidence": 0.75,
    "reasoning": "Detailed explanation of why this pair is/isn't marginally decidable, focusing on vocabulary complexity differences",
    "complexity_difference": "Description of the specific vocabulary differences between passages",
    "estimated_annotation_time": 45.0
}}"""
                ),
            ]
        )

        parser = JsonOutputParser(pydantic_object=MarginabilityAssessment)
        chain = marginality_prompt | self.llm | parser

        # Apply rate limiting
        await self._rate_limit()

        try:
            logger.debug(
                f"Assessing marginality for {passage_a.segment_id} vs {passage_b.segment_id}"
            )
            result = await chain.ainvoke({})
            return MarginabilityAssessment(**result)

        except Exception as e:
            logger.error(
                f"Failed to assess marginality for {passage_a.segment_id} vs {passage_b.segment_id}: {e}"
            )
            return None

    def _calculate_quality_score(
        self,
        assessment: MarginabilityAssessment,
        passage_a: ProcessedPassage,
        passage_b: ProcessedPassage,
    ) -> float:
        """Calculate overall quality score for a marginal pair."""

        # Base score from AI confidence
        base_score = assessment.confidence

        # Bonus for good Flesch score difference (ideal range)
        flesch_diff = abs(passage_a.flesch_score - passage_b.flesch_score)
        ideal_range = self.marginality_config.get("flesch_difference_range", [5, 25])
        ideal_min, ideal_max = ideal_range
        ideal_center = (ideal_min + ideal_max) / 2

        # Score higher for differences near the center of ideal range
        flesch_bonus = 1.0 - abs(flesch_diff - ideal_center) / (ideal_max - ideal_min)
        flesch_bonus = max(0, flesch_bonus) * 0.1  # Up to 0.1 bonus

        # Bonus for vocabulary focus word diversity
        vocab_a = set(passage_a.vocabulary_focus_words)
        vocab_b = set(passage_b.vocabulary_focus_words)
        vocab_overlap = len(vocab_a & vocab_b) / max(len(vocab_a | vocab_b), 1)
        vocab_bonus = (1.0 - vocab_overlap) * 0.1  # Bonus for less overlap

        # Penalty for very short or very long passages
        avg_length = (len(passage_a.text.split()) + len(passage_b.text.split())) / 2
        if 40 <= avg_length <= 120:  # Ideal word count range
            length_penalty = 0
        else:
            length_penalty = min(0.1, abs(avg_length - 80) / 80 * 0.1)

        # Bonus for similar reading times (easier to compare)
        time_diff = abs(
            passage_a.estimated_reading_time - passage_b.estimated_reading_time
        )
        time_bonus = max(0, (5.0 - time_diff) / 5.0) * 0.05  # Up to 0.05 bonus

        total_score = (
            base_score + flesch_bonus + vocab_bonus - length_penalty + time_bonus
        )
        return min(1.0, max(0.0, total_score))  # Clamp to [0, 1]

    async def generate_marginal_pairs(
        self, passages: List[ProcessedPassage], target_pairs: int, resume: bool = False
    ) -> List[MarginalPair]:
        """Generate marginally decidable passage pairs with caching and resume capability."""

        logger.info(
            f"Generating {target_pairs} marginal pairs from {len(passages)} passages (resume={resume})"
        )

        # Load existing progress and cache
        if resume:
            progress = self.load_progress()
            self.processed_pairs = self.load_cached_results()
            completed_hashes = set(progress["completed_pair_hashes"])
            failed_hashes = set(progress["failed_pair_hashes"])
        else:
            progress = {
                "completed_pair_hashes": [],
                "failed_pair_hashes": [],
                "total_processed": 0,
                "start_time": time.time(),
            }
            completed_hashes = set()
            failed_hashes = set()

        # Step 1: Filter candidate pairs based on business rules
        candidates = self._filter_candidate_pairs(passages)

        if not candidates:
            logger.error("No candidate pairs found after filtering")
            return []

        # Step 2: Assess marginality with AI (with progress tracking)
        marginal_pairs = self.processed_pairs.copy()  # Start with cached results
        confidence_threshold = self.marginality_config.get("confidence_threshold", 0.6)

        logger.info(
            f"Assessing {len(candidates)} candidate pairs with AI (starting with {len(marginal_pairs)} cached)"
        )

        for i, (passage_a, passage_b) in enumerate(candidates):
            pair_hash = self._create_pair_hash(passage_a, passage_b)

            # Skip if already processed
            if pair_hash in completed_hashes or pair_hash in failed_hashes:
                logger.debug(
                    f"Skipping already processed pair {passage_a.segment_id} vs {passage_b.segment_id}"
                )
                continue
            try:
                assessment = await self._assess_marginality(passage_a, passage_b)

                if (
                    assessment
                    and assessment.is_marginal
                    and assessment.confidence >= confidence_threshold
                ):
                    # Calculate quality score
                    quality_score = self._calculate_quality_score(
                        assessment, passage_a, passage_b
                    )

                    # Create marginal pair
                    pair_id = f"pair-{len(marginal_pairs) + 1:03d}"
                    marginal_pair = MarginalPair(
                        pair_id=pair_id,
                        passage_a=passage_a,
                        passage_b=passage_b,
                        marginality_confidence=assessment.confidence,
                        complexity_difference=abs(
                            passage_a.flesch_score - passage_b.flesch_score
                        ),
                        reasoning=assessment.reasoning,
                        target_annotation_time=assessment.estimated_annotation_time,
                        pair_quality_score=quality_score,
                        generation_timestamp=pd.Timestamp.now().isoformat(),
                    )

                    marginal_pairs.append(marginal_pair)

                    logger.info(
                        f"✅ Added marginal pair {pair_id} (confidence: {assessment.confidence:.3f}, quality: {quality_score:.3f})"
                    )

                else:
                    reason = "low confidence" if assessment else "assessment failed"
                    if assessment and not assessment.is_marginal:
                        reason = "not marginal"
                    logger.debug(
                        f"❌ Rejected pair {passage_a.segment_id} vs {passage_b.segment_id}: {reason}"
                    )

                # Update progress tracking
                progress["completed_pair_hashes"].append(pair_hash)
                progress["total_processed"] += 1

                # Save progress periodically
                if progress["total_processed"] % 5 == 0:
                    self.save_progress(progress)
                    self.processed_pairs = marginal_pairs.copy()
                    self.save_cached_results(self.processed_pairs)

                logger.info(
                    f"Progress: {progress['total_processed']}/{len(candidates)} pairs assessed, {len(marginal_pairs)} marginal pairs found"
                )

                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Progress: {i + 1}/{len(candidates)} assessed, {len(marginal_pairs)} pairs found"
                    )

                # Early termination if we have enough high-quality pairs
                if len(marginal_pairs) >= target_pairs * 2:  # Get extra for selection
                    logger.info(
                        f"Found sufficient pairs ({len(marginal_pairs)}), stopping early"
                    )
                    break

            except Exception as e:
                logger.error(f"Error processing candidate pair {i}: {e}")
                progress["failed_pair_hashes"].append(pair_hash)
                continue

        # Final save
        self.save_progress(progress)
        self.processed_pairs = marginal_pairs.copy()
        self.save_cached_results(self.processed_pairs)

        # Step 3: Select best pairs by quality score
        if len(marginal_pairs) > target_pairs:
            marginal_pairs.sort(key=lambda x: x.pair_quality_score, reverse=True)
            marginal_pairs = marginal_pairs[:target_pairs]
            logger.info(f"Selected top {target_pairs} pairs by quality score")

        logger.info(f"🎉 Generated {len(marginal_pairs)} marginal pairs")
        return marginal_pairs


def load_segmented_passages(input_path: Path) -> List[ProcessedPassage]:
    """Load segmented passages from Stage 1 output."""
    # Resolve input path relative to workspace root
    resolved_input_path = resolve_workspace_path(input_path)
    logger.info(f"Loading segmented passages from {resolved_input_path}")

    try:
        with open(resolved_input_path, "r") as f:
            data = json.load(f)

        # Validate file format
        if "segmented_passages" not in data:
            raise ValueError(
                "Invalid input file format: missing 'segmented_passages' key"
            )

        passages = []
        for passage_data in data["segmented_passages"]:
            passage = ProcessedPassage(**passage_data)
            passages.append(passage)

        logger.info(f"Loaded {len(passages)} segmented passages")
        return passages

    except Exception as e:
        logger.error(f"Failed to load segmented passages: {e}")
        raise


def save_marginal_pairs(pairs: List[MarginalPair], output_path: Path) -> None:
    """Save marginal pairs to JSON file."""
    # Resolve output path relative to workspace root
    resolved_output_path = resolve_output_file(output_path)

    try:
        # Convert to serializable format
        output_data = {
            "metadata": {
                "total_pairs": len(pairs),
                "generation_timestamp": pd.Timestamp.now().isoformat(),
                "schema_version": "1.0",
                "average_confidence": (
                    sum(pair.marginality_confidence for pair in pairs) / len(pairs)
                    if pairs
                    else 0
                ),
                "average_quality_score": (
                    sum(pair.pair_quality_score for pair in pairs) / len(pairs)
                    if pairs
                    else 0
                ),
            },
            "marginal_pairs": [],
        }

        for pair in pairs:
            pair_data = {
                "pair_id": pair.pair_id,
                "passage_a": (
                    asdict(pair.passage_a)
                    if hasattr(pair.passage_a, "__dict__")
                    else pair.passage_a
                ),
                "passage_b": (
                    asdict(pair.passage_b)
                    if hasattr(pair.passage_b, "__dict__")
                    else pair.passage_b
                ),
                "marginality_confidence": pair.marginality_confidence,
                "complexity_difference": pair.complexity_difference,
                "reasoning": pair.reasoning,
                "target_annotation_time": pair.target_annotation_time,
                "pair_quality_score": pair.pair_quality_score,
                "generation_timestamp": pair.generation_timestamp,
            }
            output_data["marginal_pairs"].append(pair_data)

        # Ensure output directory exists
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(resolved_output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"✅ Saved {len(pairs)} marginal pairs to {resolved_output_path}")

    except Exception as e:
        logger.error(f"Failed to save marginal pairs: {e}")
        raise


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate marginal pairs from segmented passages"
    )
    parser.add_argument(
        "--input", required=True, help="Path to segmented passages JSON from Stage 1"
    )
    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for marginal pairs JSON"
    )
    parser.add_argument(
        "--target-pairs", type=int, help="Target number of marginal pairs to generate"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous run"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override target pairs if specified
    if args.target_pairs:
        config.setdefault("marginality", {})[
            "target_marginal_pairs"
        ] = args.target_pairs

    target_pairs = config.get("marginality", {}).get("target_marginal_pairs", 50)

    # Get API key
    config_manager = SecureConfigManager()
    api_key = config_manager.get_gemini_api_key()
    if not api_key:
        logger.error("Gemini API key not found. Run setup_bazel_env.py first.")
        return 1

    # Load segmented passages
    try:
        passages = load_segmented_passages(Path(args.input))
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return 1

    if not passages:
        logger.error("No segmented passages found in input file")
        return 1

    # Initialize generator
    generator = MarginalPairGenerator(config, api_key)
    generator.setup_caching(Path(args.output))

    try:
        # Generate marginal pairs
        marginal_pairs = await generator.generate_marginal_pairs(
            passages, target_pairs, resume=args.resume
        )

        if not marginal_pairs:
            logger.warning("No marginal pairs generated")
            return 1

        # Save results
        save_marginal_pairs(marginal_pairs, Path(args.output))

        # Clean up cache and progress files on successful completion
        if not args.resume:  # Keep files if resuming for debugging
            if generator.cache_file and generator.cache_file.exists():
                generator.cache_file.unlink()
            if generator.progress_file and generator.progress_file.exists():
                generator.progress_file.unlink()

        # Print summary statistics
        avg_confidence = sum(
            pair.marginality_confidence for pair in marginal_pairs
        ) / len(marginal_pairs)
        avg_quality = sum(pair.pair_quality_score for pair in marginal_pairs) / len(
            marginal_pairs
        )

        logger.info(
            f"""
🎉 Marginal pair generation completed successfully!

Summary:
- Generated pairs: {len(marginal_pairs)}
- Average confidence: {avg_confidence:.3f}
- Average quality score: {avg_quality:.3f}
- Output file: {args.output}
        """
        )

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
