"""Batch processing helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, List

from .pipeline import AnalysisResult, DomTreeAnalyzer


def read_urls(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".txt", ""}:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "url" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'url' column")
            return [row["url"] for row in reader if row.get("url")]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item["url"] if isinstance(item, dict) else str(item) for item in data]
        raise ValueError("JSON batch file must be a list")
    raise ValueError(f"Unsupported batch file format: {path.suffix}")


def run_batch_from_file(
    path: Path,
    analyzer: DomTreeAnalyzer,
    *,
    on_result: Callable[[AnalysisResult, int], None] | None = None,
) -> List[AnalysisResult]:
    urls = read_urls(path)
    return analyzer.run_batch(urls, on_result=on_result)
