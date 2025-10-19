"""Hierarchical F1 computation between two trees."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from ..tree import TreeNode


def _enumerate_paths(node: TreeNode, *, use_label: bool = True) -> List[Tuple[str, ...]]:
    label = node.label if use_label and node.label else node.name
    current_path = (label,)
    paths = [current_path]
    for child in node.children:
        child_paths = _enumerate_paths(child, use_label=use_label)
        for path in child_paths:
            paths.append(current_path + path)
    return paths


def _weight(path: Tuple[str, ...], mode: str) -> float:
    depth = len(path)
    if mode == "depth":
        return depth
    if mode == "inverse_depth":
        return 1 / depth
    return 1.0


def hierarchical_f1(
    reference: TreeNode,
    candidate: TreeNode,
    *,
    use_label: bool = True,
    weight_mode: str = "depth",
) -> dict:
    """Compute hierarchical precision/recall/F1 using path match semantics."""

    ref_paths = _enumerate_paths(reference, use_label=use_label)
    cand_paths = _enumerate_paths(candidate, use_label=use_label)

    ref_set = {tuple(path) for path in ref_paths}
    cand_set = {tuple(path) for path in cand_paths}

    intersection = ref_set & cand_set

    def _total_weight(paths: Iterable[Tuple[str, ...]]) -> float:
        return sum(_weight(path, weight_mode) for path in paths)

    ref_weight = _total_weight(ref_set)
    cand_weight = _total_weight(cand_set)
    match_weight = _total_weight(intersection)

    precision = match_weight / cand_weight if cand_weight else 0.0
    recall = match_weight / ref_weight if ref_weight else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": len(intersection),
        "ref_paths": len(ref_set),
        "cand_paths": len(cand_set),
    }
