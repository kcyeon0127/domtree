"""Typer-based CLI for the DOMTree analysis system."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import typer

from .batch import run_batch_from_file
from .capture import CaptureOptions
from .human_tree import HumanTreeOptions
from .llm_tree import (
    HeuristicLLMOptions,
    HeuristicLLMTreeGenerator,
    OllamaVisionLLMTreeGenerator,
    OllamaVisionOptions,
)
from .pipeline import AnalysisResult, DomTreeAnalyzer
from .reporting import export_csv

app = typer.Typer(help="Analyse differences between human-perceived and LLM-derived DOM trees.")


_CAPTURE_SETTINGS = {
    "wait_after_load": 1.0,
    "max_scroll_steps": 40,
}

_HUMAN_SETTINGS = {
    "min_text_length": 20,
}

_LLM_SETTINGS = {
    "max_depth": 4,
    "max_children": 6,
}

_OUTPUT_ROOT = Path("data/output")

_OLLAMA_ENDPOINT_DEFAULT = "http://localhost:11403/api/chat"
_OLLAMA_MODEL_DEFAULT = "llama3.2-vision:11b"


def _create_llm_generator(
    backend: str,
    *,
    min_text_length: int,
    ollama_endpoint: str,
    ollama_model: str,
):
    backend = backend.lower()
    if backend == "ollama":
        return OllamaVisionLLMTreeGenerator(
            options=OllamaVisionOptions(
                endpoint=ollama_endpoint,
                model=ollama_model,
            )
        )
    if backend == "heuristic":
        return HeuristicLLMTreeGenerator(
            options=HeuristicLLMOptions(
                max_depth=_LLM_SETTINGS["max_depth"],
                max_children=_LLM_SETTINGS["max_children"],
                human_tree_options=HumanTreeOptions(min_text_length=min_text_length),
            )
        )
    raise ValueError(f"Unsupported llm backend: {backend}")


def _create_analyzer(
    *,
    llm_backend: str,
    ollama_endpoint: str,
    ollama_model: str,
) -> DomTreeAnalyzer:
    capture_options = CaptureOptions(**_CAPTURE_SETTINGS)
    human_options = HumanTreeOptions(**_HUMAN_SETTINGS)
    llm_generator = _create_llm_generator(
        backend=llm_backend,
        min_text_length=_HUMAN_SETTINGS["min_text_length"],
        ollama_endpoint=ollama_endpoint,
        ollama_model=ollama_model,
    )
    return DomTreeAnalyzer(
        capture_options=capture_options,
        human_options=human_options,
        llm_generator=llm_generator,
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"https?://", "", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("_")
    return cleaned or "analysis"


def _prepare_run_dir(category: str, slug: str) -> Path:
    run_dir = _OUTPUT_ROOT / category / slug / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_analysis(analyzer: DomTreeAnalyzer, result: AnalysisResult, run_dir: Path) -> None:
    record = result.to_dict()
    _write_json(run_dir / "result.json", record)
    (run_dir / "human_tree.json").write_text(result.human_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "llm_tree.json").write_text(result.llm_tree.to_json(indent=2), encoding="utf-8")
    analyzer.visualize(
        result,
        side_by_side_path=run_dir / "comparison.png",
        human_path=run_dir / "human.png",
        llm_path=run_dir / "llm.png",
    )


def _save_batch(results: Iterable[AnalysisResult], summary: dict, run_dir: Path) -> None:
    result_list = list(results)
    detailed_records = [result.to_dict() for result in result_list]
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "results.json", detailed_records)
    export_csv(result_list, run_dir / "results.csv")


@app.command()
def analyze(
    url: str = typer.Argument(..., help="Target URL"),
    llm_backend: str = typer.Option("ollama", "--llm-backend", help="LLM backend: ollama or heuristic"),
    ollama_endpoint: str = typer.Option(
        _OLLAMA_ENDPOINT_DEFAULT,
        "--ollama-endpoint",
        help="Ollama chat endpoint",
    ),
    ollama_model: str = typer.Option(
        _OLLAMA_MODEL_DEFAULT,
        "--ollama-model",
        help="Ollama model identifier",
    ),
) -> None:
    """Run the full pipeline for a single URL and persist outputs."""

    analyzer = _create_analyzer(
        llm_backend=llm_backend,
        ollama_endpoint=ollama_endpoint,
        ollama_model=ollama_model,
    )
    result = analyzer.analyze_url(url)
    slug = _slugify(url)
    run_dir = _prepare_run_dir("single", slug)
    _save_analysis(analyzer, result, run_dir)


@app.command("analyze-offline")
def analyze_offline(
    html_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    screenshot_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    identifier: Optional[str] = typer.Argument(None, help="Optional identifier for saved outputs"),
    llm_backend: str = typer.Option("ollama", "--llm-backend"),
    ollama_endpoint: str = typer.Option(_OLLAMA_ENDPOINT_DEFAULT, "--ollama-endpoint"),
    ollama_model: str = typer.Option(_OLLAMA_MODEL_DEFAULT, "--ollama-model"),
) -> None:
    """Run analysis for pre-downloaded HTML and screenshot files."""

    analyzer = DomTreeAnalyzer(
        human_options=HumanTreeOptions(**_HUMAN_SETTINGS),
        llm_generator=_create_llm_generator(
            backend=llm_backend,
            min_text_length=_HUMAN_SETTINGS["min_text_length"],
            ollama_endpoint=ollama_endpoint,
            ollama_model=ollama_model,
        ),
    )
    label = identifier or html_path.stem
    result = analyzer.analyze_offline(html_path=html_path, screenshot_path=screenshot_path, url=label)
    slug = _slugify(label)
    run_dir = _prepare_run_dir("offline", slug)
    _save_analysis(analyzer, result, run_dir)


@app.command()
def batch(
    batch_file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Text/CSV/JSON list of URLs"),
    identifier: Optional[str] = typer.Argument(None, help="Optional identifier for saved outputs"),
    llm_backend: str = typer.Option("ollama", "--llm-backend"),
    ollama_endpoint: str = typer.Option(_OLLAMA_ENDPOINT_DEFAULT, "--ollama-endpoint"),
    ollama_model: str = typer.Option(_OLLAMA_MODEL_DEFAULT, "--ollama-model"),
) -> None:
    """Run the pipeline on a batch of URLs and persist summary/records."""

    analyzer = _create_analyzer(
        llm_backend=llm_backend,
        ollama_endpoint=ollama_endpoint,
        ollama_model=ollama_model,
    )
    results = run_batch_from_file(batch_file, analyzer)
    summary = analyzer.summarize(results)
    label = identifier or batch_file.stem
    slug = _slugify(label)
    run_dir = _prepare_run_dir("batch", slug)
    _save_batch(results, summary, run_dir)


if __name__ == "__main__":  # pragma: no cover
    app()
