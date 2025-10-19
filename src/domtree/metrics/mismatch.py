"""Categorise mismatch patterns between two trees."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from ..tree import TreeNode
from .reading_order import reading_order_alignment


def _path_listing(node: TreeNode, *, use_label: bool = True) -> List[Tuple[Tuple[str, ...], int]]:
    entries: List[Tuple[Tuple[str, ...], int]] = []

    def _dfs(current: TreeNode, prefix: Tuple[str, ...]):
        label = current.label if use_label and current.label else current.name
        path = prefix + (label,)
        entries.append((path, len(path)))
        for child in current.children:
            _dfs(child, path)

    _dfs(node, tuple())
    return entries


def classify_mismatch_patterns(reference: TreeNode, candidate: TreeNode, *, max_examples: int = 5) -> Dict[str, Dict]:
    ref_paths = _path_listing(reference)
    cand_paths = _path_listing(candidate)

    ref_set = {path for path, _ in ref_paths}
    cand_set = {path for path, _ in cand_paths}

    missing = [path for path in ref_set - cand_set]
    extra = [path for path in cand_set - ref_set]

    # Depth shift: nodes with same tail node but different depth
    ref_depths = defaultdict(list)
    cand_depths = defaultdict(list)
    for path, depth in ref_paths:
        ref_depths[path[-1]].append(depth)
    for path, depth in cand_paths:
        cand_depths[path[-1]].append(depth)

    common_labels = set(ref_depths) & set(cand_depths)
    depth_shift_stats = []
    for label in common_labels:
        avg_ref = sum(ref_depths[label]) / len(ref_depths[label])
        avg_cand = sum(cand_depths[label]) / len(cand_depths[label])
        shift = avg_cand - avg_ref
        if abs(shift) >= 1:
            depth_shift_stats.append((label, shift))

    reading_alignment = reading_order_alignment(reference, candidate)
    order_mismatches = sum(1 for ref, cand in reading_alignment.alignment if ref is None or cand is None)

    label_counter_missing = Counter(path[-1] for path in missing)
    label_counter_extra = Counter(path[-1] for path in extra)

    return {
        "missing_nodes": {
            "count": len(missing),
            "examples": missing[:max_examples],
            "top_labels": label_counter_missing.most_common(max_examples),
        },
        "extra_nodes": {
            "count": len(extra),
            "examples": extra[:max_examples],
            "top_labels": label_counter_extra.most_common(max_examples),
        },
        "depth_shift": {
            "count": len(depth_shift_stats),
            "examples": depth_shift_stats[:max_examples],
        },
        "reading_order": {
            "gaps": order_mismatches,
            "normalized_score": reading_alignment.normalized_score,
        },
    }
