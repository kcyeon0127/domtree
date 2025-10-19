"""Structural similarity metrics."""

from __future__ import annotations

from collections import Counter
from typing import Dict

from ..tree import TreeNode, tree_depth, tree_size


def _average_branching(node: TreeNode) -> float:
    total_children = 0
    total_nodes = 0
    for current in node.traverse():
        total_nodes += 1
        total_children += len(current.children)
    return total_children / total_nodes if total_nodes else 0.0


def _collect_node_names(node: TreeNode) -> Counter:
    counter = Counter()
    for current in node.traverse():
        counter[current.name] += 1
    return counter


def _cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    all_keys = set(counter_a) | set(counter_b)
    if not all_keys:
        return 1.0
    dot = sum(counter_a[key] * counter_b[key] for key in all_keys)
    norm_a = sum(value * value for value in counter_a.values()) ** 0.5
    norm_b = sum(value * value for value in counter_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def structural_similarity(reference: TreeNode, candidate: TreeNode) -> Dict[str, float]:
    ref_size = tree_size(reference)
    cand_size = tree_size(candidate)
    size_similarity = 1 - abs(ref_size - cand_size) / max(ref_size, cand_size, 1)

    ref_depth = tree_depth(reference)
    cand_depth = tree_depth(candidate)
    depth_similarity = 1 - abs(ref_depth - cand_depth) / max(ref_depth, cand_depth, 1)

    ref_branch = _average_branching(reference)
    cand_branch = _average_branching(candidate)
    branch_similarity = 1 - abs(ref_branch - cand_branch) / max(ref_branch, cand_branch, 1)

    name_similarity = _cosine_similarity(_collect_node_names(reference), _collect_node_names(candidate))

    components = {
        "size": round(max(size_similarity, 0.0), 4),
        "depth": round(max(depth_similarity, 0.0), 4),
        "branching": round(max(branch_similarity, 0.0), 4),
        "label_distribution": round(max(name_similarity, 0.0), 4),
    }
    overall = sum(components.values()) / len(components)
    components["overall"] = round(overall, 4)
    return components
