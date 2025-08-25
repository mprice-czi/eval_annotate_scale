#!/usr/bin/env python3
"""
Statistical Pair Filtering Module

This module provides ML-based pre-filtering for marginal pair generation,
dramatically reducing the number of pairs that need AI assessment by using
sklearn to identify the most promising candidates.

Key Features:
- TF-IDF vectorization for vocabulary analysis
- Feature extraction (Flesch scores, complexity metrics, etc.)
- Clustering for strategic sampling
- Boundary detection between complexity groups
- Diversity sampling to avoid redundant pairs
"""

import logging
import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class ProcessedPassage:
    """Represents a processed passage segment (matching the main module)."""
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


class StatisticalPairFilter:
    """ML-based pre-filtering for marginal pair generation."""

    def __init__(self, config: Dict):
        """Initialize the statistical filter."""
        self.config = config
        self.marginality_config = config.get("marginality", {})
        self.pairing_config = config.get("pairing", {})
        self.statistical_config = config.get("statistical_filtering", {})
        
        # ML components with configurable parameters
        max_features = self.statistical_config.get("max_tfidf_features", 1000)
        n_clusters = self.statistical_config.get("n_clusters", 6)
        pca_components = self.statistical_config.get("pca_components", 10)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Allow single occurrences for small datasets
            max_df=0.95  # Less aggressive filtering for small datasets
        )
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        self.pca = PCA(n_components=pca_components, random_state=42)
        
        # Fitted state
        self.is_fitted = False
        self.passage_features: np.ndarray = None
        self.cluster_labels: np.ndarray = None
        self.passage_indices: Dict[str, int] = {}

    def _calculate_word_complexity_score(self, passage: ProcessedPassage) -> float:
        """Calculate a complexity score based on word characteristics."""
        words = passage.text.split()
        if not words:
            return 0.0
            
        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:"()[]')) for word in words) / len(words)
        
        # Syllable estimation (rough)
        def count_syllables(word):
            word = word.lower().strip('.,!?;:"()')
            vowels = 'aeiouy'
            count = 0
            prev_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel
            return max(1, count)
        
        avg_syllables = sum(count_syllables(word) for word in words) / len(words)
        
        # Combine metrics
        return (avg_word_length * 0.4) + (avg_syllables * 0.6)

    def _calculate_sentence_complexity(self, passage: ProcessedPassage) -> float:
        """Calculate sentence structure complexity."""
        sentences = re.split(r'[.!?]+', passage.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
            
        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Subordinate clause indicators (rough estimate)
        subordinate_indicators = ['that', 'which', 'who', 'when', 'where', 'why', 'because', 'although', 'if']
        subordinate_count = sum(passage.text.lower().count(indicator) for indicator in subordinate_indicators)
        subordinate_ratio = subordinate_count / len(passage.text.split()) if passage.text.split() else 0
        
        return (avg_sentence_length * 0.7) + (subordinate_ratio * 100 * 0.3)

    def _extract_features(self, passages: List[ProcessedPassage]) -> np.ndarray:
        """Extract comprehensive features for statistical analysis."""
        logger.info(f"Extracting features from {len(passages)} passages")
        
        # Extract text features
        texts = [passage.text for passage in passages]
        tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        
        # Extract numerical features
        numerical_features = []
        for passage in passages:
            features = [
                passage.flesch_score,
                len(passage.text.split()),  # Word count
                len(passage.vocabulary_focus_words),
                passage.estimated_reading_time,
                self._calculate_word_complexity_score(passage),
                self._calculate_sentence_complexity(passage),
                int(passage.context_preserved),  # Boolean to int
                len(set(passage.text.lower().split())),  # Unique word count
            ]
            numerical_features.append(features)
        
        # Normalize numerical features
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        combined_features = np.hstack([numerical_features, tfidf_features])
        
        # Apply PCA for dimensionality reduction if features are too large
        if combined_features.shape[1] > 100:
            combined_features = self.pca.fit_transform(combined_features)
            logger.info(f"Applied PCA: {tfidf_features.shape[1]} -> {combined_features.shape[1]} dimensions")
        
        return combined_features

    def fit(self, passages: List[ProcessedPassage]) -> None:
        """Fit the statistical models on the passage data."""
        logger.info(f"Fitting statistical models on {len(passages)} passages")
        
        # Create passage index mapping
        self.passage_indices = {passage.segment_id: i for i, passage in enumerate(passages)}
        
        # Extract features
        self.passage_features = self._extract_features(passages)
        
        # Cluster passages
        self.cluster_labels = self.clusterer.fit_predict(self.passage_features)
        
        self.is_fitted = True
        logger.info(f"Clustering found {len(np.unique(self.cluster_labels))} clusters")

    def _find_cluster_boundary_pairs(
        self, 
        passages: List[ProcessedPassage], 
        max_pairs: int = 300
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Find promising pairs near cluster boundaries."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before finding boundary pairs")
        
        boundary_pairs = []
        seen_pairs = set()
        
        # For each cluster, find passages near the boundary with other clusters
        for cluster_a in range(self.clusterer.n_clusters):
            for cluster_b in range(cluster_a + 1, self.clusterer.n_clusters):
                # Get passages from each cluster
                passages_a = [p for i, p in enumerate(passages) if self.cluster_labels[i] == cluster_a]
                passages_b = [p for i, p in enumerate(passages) if self.cluster_labels[i] == cluster_b]
                
                if not passages_a or not passages_b:
                    continue
                
                # Find closest pairs between clusters (potential boundaries)
                for passage_a in passages_a[:10]:  # Limit for efficiency
                    for passage_b in passages_b[:10]:
                        pair_key = tuple(sorted([passage_a.segment_id, passage_b.segment_id]))
                        if pair_key in seen_pairs:
                            continue
                        
                        # Check if they meet basic criteria
                        if self._meets_basic_criteria(passage_a, passage_b):
                            boundary_pairs.append((passage_a, passage_b))
                            seen_pairs.add(pair_key)
                            
                            if len(boundary_pairs) >= max_pairs:
                                logger.info(f"Found {len(boundary_pairs)} cluster boundary pairs")
                                return boundary_pairs
        
        logger.info(f"Found {len(boundary_pairs)} cluster boundary pairs")
        return boundary_pairs

    def _find_marginal_pairs_within_clusters(
        self,
        passages: List[ProcessedPassage],
        max_pairs: int = 200
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Find marginal pairs within the same cluster (similar complexity)."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before finding marginal pairs")
        
        marginal_pairs = []
        seen_pairs = set()
        
        for cluster_id in range(self.clusterer.n_clusters):
            cluster_passages = [p for i, p in enumerate(passages) if self.cluster_labels[i] == cluster_id]
            
            if len(cluster_passages) < 2:
                continue
            
            # Within cluster, find pairs with intermediate differences
            flesch_range = self.marginality_config.get("flesch_difference_range", [5, 25])
            min_diff, max_diff = flesch_range
            
            for i, passage_a in enumerate(cluster_passages):
                for passage_b in cluster_passages[i+1:]:
                    pair_key = tuple(sorted([passage_a.segment_id, passage_b.segment_id]))
                    if pair_key in seen_pairs:
                        continue
                        
                    # Check Flesch score difference is in marginal range
                    flesch_diff = abs(passage_a.flesch_score - passage_b.flesch_score)
                    if min_diff <= flesch_diff <= max_diff:
                        if self._meets_basic_criteria(passage_a, passage_b):
                            marginal_pairs.append((passage_a, passage_b))
                            seen_pairs.add(pair_key)
                            
                            if len(marginal_pairs) >= max_pairs:
                                logger.info(f"Found {len(marginal_pairs)} marginal pairs within clusters")
                                return marginal_pairs
        
        logger.info(f"Found {len(marginal_pairs)} marginal pairs within clusters")
        return marginal_pairs

    def _diversity_sampling(
        self,
        passages: List[ProcessedPassage],
        max_pairs: int = 500
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Use diversity sampling to find varied, interesting pairs."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before diversity sampling")
        
        diverse_pairs = []
        seen_pairs = set()
        
        # Calculate pairwise distances in feature space
        distances = euclidean_distances(self.passage_features)
        
        # Find pairs with intermediate distances (not too similar, not too different)
        distance_percentiles = np.percentile(distances, [30, 70])
        min_dist, max_dist = distance_percentiles
        
        for i, passage_a in enumerate(passages):
            for j, passage_b in enumerate(passages[i+1:], i+1):
                pair_key = tuple(sorted([passage_a.segment_id, passage_b.segment_id]))
                if pair_key in seen_pairs:
                    continue
                
                # Check if distance is in the interesting range
                pair_distance = distances[i, j]
                if min_dist <= pair_distance <= max_dist:
                    if self._meets_basic_criteria(passage_a, passage_b):
                        diverse_pairs.append((passage_a, passage_b))
                        seen_pairs.add(pair_key)
                        
                        if len(diverse_pairs) >= max_pairs:
                            logger.info(f"Found {len(diverse_pairs)} diverse pairs")
                            return diverse_pairs
        
        logger.info(f"Found {len(diverse_pairs)} diverse pairs")
        return diverse_pairs

    def _meets_basic_criteria(
        self, 
        passage_a: ProcessedPassage, 
        passage_b: ProcessedPassage
    ) -> bool:
        """Check if a pair meets basic filtering criteria."""
        # Exclude same original document
        exclude_same_original = self.pairing_config.get("exclude_same_original", True)
        if exclude_same_original and passage_a.original_id == passage_b.original_id:
            return False
        
        # Reading time similarity
        time_diff = abs(passage_a.estimated_reading_time - passage_b.estimated_reading_time)
        if time_diff > 10:  # Within 10 seconds
            return False
        
        # Flesch score difference constraints
        flesch_range = self.marginality_config.get("flesch_difference_range", [5, 25])
        min_diff, max_diff = flesch_range
        flesch_diff = abs(passage_a.flesch_score - passage_b.flesch_score)
        if not (min_diff <= flesch_diff <= max_diff):
            return False
        
        return True

    def _calculate_marginality_likelihood(
        self,
        pair: Tuple[ProcessedPassage, ProcessedPassage]
    ) -> float:
        """Calculate likelihood that a pair is marginally decidable."""
        passage_a, passage_b = pair
        
        if not self.is_fitted:
            return 0.5  # Default score if not fitted
        
        # Get feature vectors
        idx_a = self.passage_indices.get(passage_a.segment_id)
        idx_b = self.passage_indices.get(passage_b.segment_id)
        
        if idx_a is None or idx_b is None:
            return 0.5
        
        features_a = self.passage_features[idx_a]
        features_b = self.passage_features[idx_b]
        
        # Calculate various similarity metrics
        cosine_sim = cosine_similarity([features_a], [features_b])[0, 0]
        euclidean_dist = euclidean_distances([features_a], [features_b])[0, 0]
        
        # Flesch score difference (normalized)
        flesch_diff = abs(passage_a.flesch_score - passage_b.flesch_score)
        flesch_range = self.marginality_config.get("flesch_difference_range", [5, 25])
        ideal_flesch = (flesch_range[0] + flesch_range[1]) / 2
        flesch_score = 1.0 - abs(flesch_diff - ideal_flesch) / ideal_flesch
        
        # Vocabulary diversity
        vocab_a = set(passage_a.vocabulary_focus_words)
        vocab_b = set(passage_b.vocabulary_focus_words)
        vocab_overlap = len(vocab_a & vocab_b) / max(len(vocab_a | vocab_b), 1)
        vocab_diversity = 1.0 - vocab_overlap
        
        # Combine scores (weights tuned for marginality)
        marginality_score = (
            0.3 * (1.0 - cosine_sim) +     # Some difference but not too much
            0.2 * min(euclidean_dist / np.std(euclidean_distances(self.passage_features)), 1.0) +
            0.3 * flesch_score +            # Good Flesch difference
            0.2 * vocab_diversity           # Different vocabulary
        )
        
        return max(0.0, min(1.0, marginality_score))

    def filter_candidate_pairs(
        self,
        passages: List[ProcessedPassage],
        max_candidates: int = 1000
    ) -> List[Tuple[ProcessedPassage, ProcessedPassage]]:
        """Main method: intelligently filter to most promising pairs."""
        logger.info(f"Statistical filtering of {len(passages)} passages to {max_candidates} pairs")
        
        # Fit models
        self.fit(passages)
        
        # Strategic sampling with configurable ratios
        boundary_ratio = self.statistical_config.get("boundary_pairs_ratio", 0.33)
        marginal_ratio = self.statistical_config.get("marginal_pairs_ratio", 0.33)
        diverse_ratio = self.statistical_config.get("diverse_pairs_ratio", 0.34)
        
        boundary_pairs = self._find_cluster_boundary_pairs(passages, max_pairs=int(max_candidates * boundary_ratio))
        marginal_pairs = self._find_marginal_pairs_within_clusters(passages, max_pairs=int(max_candidates * marginal_ratio))
        diverse_pairs = self._diversity_sampling(passages, max_pairs=int(max_candidates * diverse_ratio))
        
        # Combine all candidates
        all_candidates = boundary_pairs + marginal_pairs + diverse_pairs
        
        # Remove duplicates
        seen_pairs = set()
        unique_candidates = []
        for pair in all_candidates:
            pair_key = tuple(sorted([pair[0].segment_id, pair[1].segment_id]))
            if pair_key not in seen_pairs:
                unique_candidates.append(pair)
                seen_pairs.add(pair_key)
        
        # Score and rank by marginality likelihood
        scored_candidates = [
            (pair, self._calculate_marginality_likelihood(pair))
            for pair in unique_candidates
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        final_candidates = [pair for pair, score in scored_candidates[:max_candidates]]
        
        logger.info(f"Statistical pre-filtering selected {len(final_candidates)} promising pairs")
        logger.info(f"Reduction ratio: {len(passages)*(len(passages)-1)//2} -> {len(final_candidates)} ({len(final_candidates)/(len(passages)*(len(passages)-1)//2)*100:.2f}%)")
        
        return final_candidates