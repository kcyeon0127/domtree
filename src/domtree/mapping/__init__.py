"""Category mapping utilities for aligning DOM nodes with document layout labels."""

from .categories import CATEGORY_ALIASES, CATEGORY_DEFINITIONS
from .mapper import map_node_to_category, annotate_tree_with_categories
from .stats import compute_category_stats, merge_category_stats

__all__ = [
    "CATEGORY_ALIASES",
    "CATEGORY_DEFINITIONS",
    "map_node_to_category",
    "annotate_tree_with_categories",
    "compute_category_stats",
    "merge_category_stats",
]
