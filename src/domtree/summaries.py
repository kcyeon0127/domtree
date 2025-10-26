"""Utilities for computing aggregate summaries from stored metrics."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List

Record = Dict[str, object]
Metric = Dict[str, float]


def summarize_records(records: Iterable[Record]) -> dict:
    """Aggregate metric dictionaries into the summary structure used by batch runs."""

    metrics_map: Dict[str, List[Metric]] = defaultdict(list)
    mismatch_totals: Dict[str, Dict[str, int]] = defaultdict(_empty_mismatch_totals)

    for record in records:
        metrics_section = record.get("metrics") or {}
        mismatch_section = record.get("mismatch_patterns") or {}
        if not isinstance(metrics_section, dict):
            continue

        for key, metrics in metrics_section.items():
            if not isinstance(metrics, dict):
                continue
            metrics_map[key].append(metrics)
            patterns = mismatch_section.get(key)
            if patterns:
                _accumulate_mismatches(mismatch_totals[key], patterns)
            else:
                # Ensure the key exists even when mismatch details are missing.
                mismatch_totals.setdefault(key, _empty_mismatch_totals())

    summary: Dict[str, dict] = {}
    for key, metric_list in metrics_map.items():
        if not metric_list:
            continue
        summary[key] = {
            "average_metrics": _average_metrics(metric_list),
            "mismatch_totals": deepcopy(mismatch_totals.get(key, _empty_mismatch_totals())),
            "count": len(metric_list),
        }

    return summary


def _average_metrics(metrics_list: List[Metric]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for metric in metrics_list:
        for key, value in metric.items():
            if value is None:
                continue
            totals[key] += value
            counts[key] += 1
    return {key: totals[key] / counts[key] for key in totals}


def _accumulate_mismatches(target: Dict[str, int], patterns: dict) -> None:
    missing = patterns.get("missing_nodes", {})
    extra = patterns.get("extra_nodes", {})
    depth_shift = patterns.get("depth_shift", {})
    order = patterns.get("reading_order", {})
    target["missing"] += missing.get("count", 0)
    target["extra"] += extra.get("count", 0)
    target["depth_shift"] += depth_shift.get("count", 0)
    target["order"] += order.get("gaps", 0)


def _empty_mismatch_totals() -> Dict[str, int]:
    return {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}

