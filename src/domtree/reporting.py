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
