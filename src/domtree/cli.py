"""Typer-based CLI for the DOMTree analysis system."""

from __future__ import annotations

import json
import shutil
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
    OllamaVisionDomLLMTreeGenerator,
    OllamaVisionDomOptions,
    OllamaVisionLLMTreeGenerator,
    OllamaVisionOptions,
    OpenRouterVisionDomLLMTreeGenerator,
    OpenRouterVisionDomOptions,
    OpenRouterVisionLLMTreeGenerator,
    OpenRouterVisionOptions,
    OpenRouterVisionHtmlLLMTreeGenerator,
    OpenRouterVisionHtmlOptions,
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
    "restrict_to_viewport": True,
}

_LLM_SETTINGS = {
    "max_depth": 4,
    "max_children": 6,
}

_OUTPUT_ROOT = Path("data/output")

_LLM_BACKEND = "openrouter"  # options: "ollama"(llamavision), "openrouter"(gpt4omini), or "heuristic"
_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.2-vision:11b"
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = "openai/gpt-4o-mini"
_OPENROUTER_REFERER = ""
_OPENROUTER_TITLE = "DOMTree Analyzer"


def _create_llm_generators(min_text_length: int):
    backend = _LLM_BACKEND.lower()
    if backend == "ollama":
        vision = OllamaVisionLLMTreeGenerator(
            options=OllamaVisionOptions(
                endpoint=_OLLAMA_ENDPOINT,
                model=_OLLAMA_MODEL,
            )
        )
        dom = OllamaVisionDomLLMTreeGenerator(
            options=OllamaVisionDomOptions(
                endpoint=_OLLAMA_ENDPOINT,
                model=_OLLAMA_MODEL,
            )
        )
        return vision, dom, None, None
    if backend == "openrouter":
        vision = OpenRouterVisionLLMTreeGenerator(
            options=OpenRouterVisionOptions(
                endpoint=_OPENROUTER_ENDPOINT,
                model=_OPENROUTER_MODEL,
                referer=_OPENROUTER_REFERER,
                title=_OPENROUTER_TITLE,
            )
        )
        dom = OpenRouterVisionDomLLMTreeGenerator(
            options=OpenRouterVisionDomOptions(
                endpoint=_OPENROUTER_ENDPOINT,
                model=_OPENROUTER_MODEL,
                referer=_OPENROUTER_REFERER,
                title=_OPENROUTER_TITLE,
            )
        )
        html = OpenRouterVisionHtmlLLMTreeGenerator(
            options=OpenRouterVisionHtmlOptions(
                endpoint=_OPENROUTER_ENDPOINT,
                model=_OPENROUTER_MODEL,
                referer=_OPENROUTER_REFERER,
                title=_OPENROUTER_TITLE,
            )
        )
        full = OpenRouterVisionFullLLMTreeGenerator(
            options=OpenRouterVisionFullOptions(
                endpoint=_OPENROUTER_ENDPOINT,
                model=_OPENROUTER_MODEL,
                referer=_OPENROUTER_REFERER,
                title=_OPENROUTER_TITLE,
            )
        )
        return vision, dom, html, full
    if backend == "heuristic":
        generator = HeuristicLLMTreeGenerator(
            options=HeuristicLLMOptions(
                max_depth=_LLM_SETTINGS["max_depth"],
                max_children=_LLM_SETTINGS["max_children"],
                human_tree_options=HumanTreeOptions(min_text_length=min_text_length),
            )
        )
        return generator, None, None, None
    raise ValueError(f"Unsupported llm backend: {backend}")


def _create_analyzer() -> DomTreeAnalyzer:
    capture_options = CaptureOptions(**_CAPTURE_SETTINGS)
    human_options = HumanTreeOptions(**_HUMAN_SETTINGS)
    llm_generator, dom_llm_generator, html_llm_generator, full_llm_generator = _create_llm_generators(
        min_text_length=_HUMAN_SETTINGS["min_text_length"]
    )
    return DomTreeAnalyzer(
        capture_options=capture_options,
        human_options=human_options,
        llm_generator=llm_generator,
        dom_llm_generator=dom_llm_generator,
        html_llm_generator=html_llm_generator,
        full_llm_generator=full_llm_generator,
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
    (run_dir / "human_zone_tree.json").write_text(result.human_zone_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "human_heading_tree.json").write_text(result.human_heading_tree.to_json(indent=2), encoding="utf-8")
    # Backwards compatibility: keep legacy filename pointing to zone tree
    (run_dir / "human_tree.json").write_text(result.human_zone_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "llm_tree.json").write_text(result.llm_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_dom_tree is not None:
        (run_dir / "llm_dom_tree.json").write_text(result.llm_dom_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_html_tree is not None:
        (run_dir / "llm_html_tree.json").write_text(result.llm_html_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_full_tree is not None:
        (run_dir / "llm_full_tree.json").write_text(result.llm_full_tree.to_json(indent=2), encoding="utf-8")

    _write_llm_comparison(result, run_dir)
    analyzer.visualize(
        result,
        zone_side_by_side_path=run_dir / "comparison_zone.png",
        heading_side_by_side_path=run_dir / "comparison_heading.png",
        zone_path=run_dir / "human_zone.png",
        heading_path=run_dir / "human_heading.png",
        llm_path=run_dir / "llm.png",
        zone_dom_side_by_side_path=run_dir / "comparison_zone_dom.png",
        heading_dom_side_by_side_path=run_dir / "comparison_heading_dom.png",
        llm_dom_path=run_dir / "llm_dom.png",
        zone_html_side_by_side_path=run_dir / "comparison_zone_html.png",
        heading_html_side_by_side_path=run_dir / "comparison_heading_html.png",
        llm_html_path=run_dir / "llm_html.png",
        zone_full_side_by_side_path=run_dir / "comparison_zone_full.png",
        heading_full_side_by_side_path=run_dir / "comparison_heading_full.png",
        llm_full_path=run_dir / "llm_full.png",
    )
    # Legacy filenames for downstream compatibility
    zone_comparison = run_dir / "comparison_zone.png"
    zone_human = run_dir / "human_zone.png"
    if zone_comparison.exists():
        shutil.copyfile(zone_comparison, run_dir / "comparison.png")
    if zone_human.exists():
        shutil.copyfile(zone_human, run_dir / "human.png")


def _save_batch(results: Iterable[AnalysisResult], summary: dict, run_dir: Path) -> None:
    result_list = list(results)
    detailed_records = [result.to_dict() for result in result_list]
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "results.json", detailed_records)
    export_csv(result_list, run_dir / "results.csv")


def _write_llm_comparison(result: AnalysisResult, run_dir: Path) -> None:
    zone_vis = result.zone_comparison.metrics.flat()
    heading_vis = result.heading_comparison.metrics.flat()

    def _delta(candidate: dict, reference: dict) -> dict:
        delta = {}
        for key, cand_value in candidate.items():
            ref_value = reference.get(key)
            if isinstance(cand_value, (int, float)) and isinstance(ref_value, (int, float)):
                delta[key] = cand_value - ref_value
        return delta

    comparison = {
        "zone": {"vision": zone_vis},
        "heading": {"vision": heading_vis},
    }

    variants = [
        ("dom", result.zone_dom_comparison, result.heading_dom_comparison),
        ("html", result.zone_html_comparison, result.heading_html_comparison),
        ("full", result.zone_full_comparison, result.heading_full_comparison),
    ]

    for key, zone_variant, heading_variant in variants:
        if not (zone_variant and heading_variant):
            continue
        zone_metrics = zone_variant.metrics.flat()
        heading_metrics = heading_variant.metrics.flat()
        comparison["zone"][f"vision_{key}"] = zone_metrics
        comparison["zone"][f"delta_{key}"] = _delta(zone_metrics, zone_vis)
        comparison["heading"][f"vision_{key}"] = heading_metrics
        comparison["heading"][f"delta_{key}"] = _delta(heading_metrics, heading_vis)

    if len(comparison["zone"]) == 1 and len(comparison["heading"]) == 1:
        return

    _write_json(run_dir / "llm_comparison.json", comparison)


@app.command()
def analyze(url: str = typer.Argument(..., help="Target URL")) -> None:
    """Run the full pipeline for a single URL and persist outputs."""

    analyzer = _create_analyzer()
    result = analyzer.analyze_url(url)
    slug = _slugify(url)
    run_dir = _prepare_run_dir("single", slug)
    _save_analysis(analyzer, result, run_dir)


@app.command("analyze-offline")
def analyze_offline(
    html_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    screenshot_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    identifier: Optional[str] = typer.Argument(None, help="Optional identifier for saved outputs"),
) -> None:
    """Run analysis for pre-downloaded HTML and screenshot files."""

    llm_generator, dom_llm_generator, html_llm_generator, full_llm_generator = _create_llm_generators(
        min_text_length=_HUMAN_SETTINGS["min_text_length"]
    )
    analyzer = DomTreeAnalyzer(
        human_options=HumanTreeOptions(**_HUMAN_SETTINGS),
        llm_generator=llm_generator,
        dom_llm_generator=dom_llm_generator,
        html_llm_generator=html_llm_generator,
        full_llm_generator=full_llm_generator,
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
) -> None:
    """Run the pipeline on a batch of URLs and persist summary/records."""

    analyzer = _create_analyzer()
    results = run_batch_from_file(batch_file, analyzer)
    summary = analyzer.summarize(results)
    label = identifier or batch_file.stem
    slug = _slugify(label)
    run_dir = _prepare_run_dir("batch", slug)
    _save_batch(results, summary, run_dir)


if __name__ == "__main__":  # pragma: no cover
    app()
