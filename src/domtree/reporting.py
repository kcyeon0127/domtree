"""Reporting helpers: CSV export, summary tables, etc."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .pipeline import AnalysisResult


def results_to_records(results: Iterable[AnalysisResult]) -> List[dict]:
    return [result.to_dict() for result in results]


def results_to_dataframe(results: Iterable[AnalysisResult]) -> pd.DataFrame:
    records = results_to_records(results)
    frame = pd.json_normalize(records)
    return frame


def export_csv(results: Iterable[AnalysisResult], path: Path) -> Path:
    frame = results_to_dataframe(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def export_json(results: Iterable[AnalysisResult], path: Path) -> Path:
    data = results_to_records(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def combine_summaries(summary_paths: List[Path]) -> Dict[str, Any]:
    combined_summary: Dict[str, Any] = {}
    first_summary_loaded = False

    for path in summary_paths:
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        
        with path.open("r", encoding="utf-8") as f:
            current_summary = json.load(f)
        
        if not first_summary_loaded:
            for key, value in current_summary.items():
                combined_summary[key] = {
                    "average_metrics": {k: 0.0 for k in value["average_metrics"].keys()},
                    "mismatch_totals": {k: 0 for k in value["mismatch_totals"].keys()},
                    "count": 0
                }
            first_summary_loaded = True

        for key, value in current_summary.items():
            if key not in combined_summary:
                # Initialize if a key is missing in the first summary but present later
                combined_summary[key] = {
                    "average_metrics": {k: 0.0 for k in value["average_metrics"].keys()},
                    "mismatch_totals": {k: 0 for k in value["mismatch_totals"].keys()},
                    "count": 0
                }

            current_count = value["count"]
            combined_summary[key]["count"] += current_count

            for metric_key, metric_value in value["average_metrics"].items():
                combined_summary[key]["average_metrics"][metric_key] += metric_value * current_count
            
            for mismatch_key, mismatch_value in value["mismatch_totals"].items():
                combined_summary[key]["mismatch_totals"][mismatch_key] += mismatch_value
    
    # Finalize averages
    for key, value in combined_summary.items():
        total_count = value["count"]
        if total_count > 0:
            for metric_key in value["average_metrics"].keys():
                combined_summary[key]["average_metrics"][metric_key] /= total_count
        else:
            for metric_key in value["average_metrics"].keys():
                combined_summary[key]["average_metrics"][metric_key] = 0.0

    return combined_summary
