"""Helpers to measure category distributions in DOM/LLM trees."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

from ..tree import TreeNode
from .mapper import map_node_to_category


@dataclass
class CategoryStats:
    count: int = 0
    area: float = 0.0

    def add(self, *, count: int = 0, area: float = 0.0) -> None:
        self.count += count
        self.area += area


def _bbox_area(node: TreeNode) -> float:
    bbox = node.metadata.visual_cues.bbox
    if not bbox:
        return 0.0
    top, left, bottom, right = bbox
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    return width * height


def compute_category_stats(root: TreeNode) -> Dict[str, dict]:
    """Return raw/normalized stats per layout category for a single tree.

    The function iterates over all nodes, maps them to canonical categories
    using :func:`map_node_to_category`, and accumulates counts as well as
    bounding-box area (when available). The result is structured as::

        {
            "per_category": {
                "BODY_TEXT": {"count": 42, "count_ratio": 0.5,
                               "area": 1234.0, "area_ratio": 0.72},
                ...
            },
            "totals": {"count": 84, "area": 1712.0}
        }
    """

    totals: Dict[str, CategoryStats] = defaultdict(CategoryStats)
    total_count = 0
    total_area = 0.0

    for node in root.traverse():
        category = map_node_to_category(node)
        if not category:
            continue
        area = _bbox_area(node)
        stats = totals[category]
        stats.add(count=1, area=area)
        total_count += 1
        total_area += area

    per_category = {}
    for key, stats in totals.items():
        entry = {
            "count": stats.count,
            "area": stats.area,
        }
        if total_count:
            entry["count_ratio"] = stats.count / total_count
        if total_area:
            entry["area_ratio"] = stats.area / total_area
        per_category[key] = entry

    return {
        "per_category": per_category,
        "totals": {"count": total_count, "area": total_area},
    }


def merge_category_stats(stats_list: Iterable[Mapping[str, dict]]) -> Dict[str, dict]:
    """Aggregate multiple ``per_category`` maps into a single distribution."""

    agg: Dict[str, CategoryStats] = defaultdict(CategoryStats)
    total_count = 0
    total_area = 0.0

    for stats in stats_list:
        for key, value in stats.items():
            count = float(value.get("count", 0))
            area = float(value.get("area", 0.0))
            agg[key].add(count=int(count), area=area)
            total_count += int(count)
            total_area += area

    per_category = {}
    for key, stats in agg.items():
        entry = {
            "count": stats.count,
            "area": stats.area,
        }
        if total_count:
            entry["count_ratio"] = stats.count / total_count
        if total_area:
            entry["area_ratio"] = stats.area / total_area
        per_category[key] = entry

    return {
        "per_category": per_category,
        "totals": {"count": total_count, "area": total_area},
    }
