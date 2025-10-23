"""Utility helpers to map DOM/LLM tree nodes on to canonical categories."""

from __future__ import annotations

from typing import Iterable

from ..tree import TreeNode
from .categories import CATEGORY_DEFINITIONS, canonical_category


def _candidate_labels(node: TreeNode) -> Iterable[str]:
    """Yield potential labels from a ``TreeNode`` in priority order."""

    metadata = node.metadata
    if metadata.text_heading:
        yield metadata.text_heading
    if metadata.role:
        yield metadata.role
    if metadata.node_type:
        yield metadata.node_type
    if node.label:
        yield node.label
    yield node.name


def map_node_to_category(node: TreeNode) -> str | None:
    """Return the canonical layout category for a given node.

    If none of the candidate labels match ``CATEGORY_ALIASES`` the function
    returns ``None`` to indicate the node should be counted as "unmapped".
    """

    for label in _candidate_labels(node):
        if not label:
            continue
        match = canonical_category(label)
        if match:
            return match
    return None


def annotate_tree_with_categories(root: TreeNode) -> None:
    """Attach canonical category IDs to ``TreeNode.attributes["layout_category"]``.

    The function modifies the tree in-place. Nodes that do not match any
    canonical category remain untouched so downstream code can treat
    ``layout_category`` as optional. This keeps the tree JSON backward
    compatible while enabling category-level metrics in later experiments.
    """

    for node in root.traverse():
        category = map_node_to_category(node)
        if category:
            node.attributes.setdefault("layout", {})["category"] = CATEGORY_DEFINITIONS[category].name
