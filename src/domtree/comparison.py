"""Comparison utilities that compute metrics between trees."""

from __future__ import annotations

import dataclasses
from typing import Dict

from .metrics import (
    classify_mismatch_patterns,
    hierarchical_f1,
    normalized_tree_edit_distance,
    reading_order_alignment,
    structural_similarity,
    tree_edit_distance,
)
from .tree import TreeNode


@dataclasses.dataclass
class ComparisonMetrics:
    tree_edit_distance: float
    normalized_tree_edit_distance: float
    hierarchical_f1: Dict[str, float]
    structural_similarity: Dict[str, float]
    reading_order: Dict[str, float]
    mismatch_patterns: Dict[str, Dict]

    def flat(self) -> Dict[str, float]:
        flat_metrics: Dict[str, float] = {
            "tree_edit_distance": self.tree_edit_distance,
            "normalized_tree_edit_distance": self.normalized_tree_edit_distance,
            "reading_order_score": self.reading_order.get("normalized_score", 0.0),
        }
        flat_metrics.update({f"hierarchical_{key}": value for key, value in self.hierarchical_f1.items()})
        flat_metrics.update({f"structural_{key}": value for key, value in self.structural_similarity.items()})
        flat_metrics.update({
            "mismatch_missing": self.mismatch_patterns["missing_nodes"]["count"],
            "mismatch_extra": self.mismatch_patterns["extra_nodes"]["count"],
            "mismatch_depth_shift": self.mismatch_patterns["depth_shift"]["count"],
            "mismatch_order_gaps": self.mismatch_patterns["reading_order"]["gaps"],
        })
        return flat_metrics


@dataclasses.dataclass
class ComparisonResult:
    human_tree: TreeNode
    llm_tree: TreeNode
    metrics: ComparisonMetrics


def compute_comparison(human_tree: TreeNode, llm_tree: TreeNode) -> ComparisonResult:
    ted = tree_edit_distance(human_tree, llm_tree)
    normalized_ted = normalized_tree_edit_distance(human_tree, llm_tree)
    h_f1 = hierarchical_f1(human_tree, llm_tree)
    structural = structural_similarity(human_tree, llm_tree)
    reading = reading_order_alignment(human_tree, llm_tree)
    mismatch = classify_mismatch_patterns(human_tree, llm_tree)
    metrics = ComparisonMetrics(
        tree_edit_distance=ted,
        normalized_tree_edit_distance=normalized_ted,
        hierarchical_f1=h_f1,
        structural_similarity=structural,
        reading_order={
            "score": reading.score,
            "normalized_score": reading.normalized_score,
        },
        mismatch_patterns=mismatch,
    )
    return ComparisonResult(human_tree=human_tree, llm_tree=llm_tree, metrics=metrics)
