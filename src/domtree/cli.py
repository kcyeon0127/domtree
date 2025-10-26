"""Typer-based CLI for the DOMTree analysis system."""

from __future__ import annotations

import json
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import typer
import logging

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
    OpenRouterVisionFullLLMTreeGenerator,
    OpenRouterVisionFullOptions,
    OpenRouterHtmlOnlyLLMTreeGenerator,
    OpenRouterHtmlOnlyOptions,
)
from .pipeline import AnalysisResult, DomTreeAnalyzer
from .tree import TreeNode
from .mapping import annotate_tree_with_categories, compute_category_stats
from .reporting import export_csv

app = typer.Typer(help="Analyse differences between human-perceived and LLM-derived DOM trees.")
logging.basicConfig(level=logging.INFO)

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

_LLM_BACKEND = "openrouter"
_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.2-vision:11b"
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = "openai/gpt-4o-mini"
_OPENROUTER_REFERER = ""
_OPENROUTER_TITLE = "DOMTree Analyzer"

def _create_llm_generators(min_text_length: int):
    backend = _LLM_BACKEND.lower()
    if backend == "ollama":
        vision = OllamaVisionLLMTreeGenerator(options=OllamaVisionOptions(endpoint=_OLLAMA_ENDPOINT, model=_OLLAMA_MODEL))
        dom = OllamaVisionDomLLMTreeGenerator(options=OllamaVisionDomOptions(endpoint=_OLLAMA_ENDPOINT, model=_OLLAMA_MODEL))
        return vision, dom, None, None, None
    if backend == "openrouter":
        vision = OpenRouterVisionLLMTreeGenerator(options=OpenRouterVisionOptions(endpoint=_OPENROUTER_ENDPOINT, model=_OPENROUTER_MODEL, referer=_OPENROUTER_REFERER, title=_OPENROUTER_TITLE))
        dom = OpenRouterVisionDomLLMTreeGenerator(options=OpenRouterVisionDomOptions(endpoint=_OPENROUTER_ENDPOINT, model=_OPENROUTER_MODEL, referer=_OPENROUTER_REFERER, title=_OPENROUTER_TITLE))
        html = OpenRouterVisionHtmlLLMTreeGenerator(options=OpenRouterVisionHtmlOptions(endpoint=_OPENROUTER_ENDPOINT, model=_OPENROUTER_MODEL, referer=_OPENROUTER_REFERER, title=_OPENROUTER_TITLE))
        html_only = OpenRouterHtmlOnlyLLMTreeGenerator(options=OpenRouterHtmlOnlyOptions(endpoint=_OPENROUTER_ENDPOINT, model=_OPENROUTER_MODEL, referer=_OPENROUTER_REFERER, title=_OPENROUTER_TITLE))
        full = OpenRouterVisionFullLLMTreeGenerator(options=OpenRouterVisionFullOptions(endpoint=_OPENROUTER_ENDPOINT, model=_OPENROUTER_MODEL, referer=_OPENROUTER_REFERER, title=_OPENROUTER_TITLE))
        return vision, dom, html, full, html_only
    if backend == "heuristic":
        generator = HeuristicLLMTreeGenerator(options=HeuristicLLMOptions(max_depth=_LLM_SETTINGS["max_depth"], max_children=_LLM_SETTINGS["max_children"], human_tree_options=HumanTreeOptions(min_text_length=min_text_length)))
        return generator, None, None, None, None
    raise ValueError(f"Unsupported llm backend: {backend}")

def _create_analyzer() -> DomTreeAnalyzer:
    capture_options = CaptureOptions(**_CAPTURE_SETTINGS)
    human_options = HumanTreeOptions(**_HUMAN_SETTINGS)
    llm_generator, dom_llm_generator, html_llm_generator, full_llm_generator, html_only_llm_generator = _create_llm_generators(min_text_length=_HUMAN_SETTINGS["min_text_length"])
    return DomTreeAnalyzer(capture_options=capture_options, human_options=human_options, llm_generator=llm_generator, dom_llm_generator=dom_llm_generator, html_llm_generator=html_llm_generator, html_only_llm_generator=html_only_llm_generator, full_llm_generator=full_llm_generator)

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
    (run_dir / "zone_tree.json").write_text(result.zone_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "heading_tree.json").write_text(result.heading_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "contraction_tree.json").write_text(result.contraction_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "llm_tree.json").write_text(result.llm_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_dom_tree:
        (run_dir / "llm_dom_tree.json").write_text(result.llm_dom_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_html_tree:
        (run_dir / "llm_html_tree.json").write_text(result.llm_html_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_full_tree:
        (run_dir / "llm_full_tree.json").write_text(result.llm_full_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_html_only_tree:
        (run_dir / "llm_html_only_tree.json").write_text(result.llm_html_only_tree.to_json(indent=2), encoding="utf-8")

    _write_llm_comparison(result, run_dir)
    analyzer.visualize(
        result,
        zone_path=run_dir / "zone.png",
        heading_path=run_dir / "heading.png",
        contraction_path=run_dir / "contraction.png",
        llm_path=run_dir / "llm.png",
        zone_side_by_side_path=run_dir / "comparison_zone.png",
        heading_side_by_side_path=run_dir / "comparison_heading.png",
        contraction_side_by_side_path=run_dir / "comparison_contraction.png",
        zone_dom_side_by_side_path=run_dir / "comparison_zone_dom.png",
        heading_dom_side_by_side_path=run_dir / "comparison_heading_dom.png",
        contraction_dom_side_by_side_path=run_dir / "comparison_contraction_dom.png",
        llm_dom_path=run_dir / "llm_dom.png",
        zone_html_side_by_side_path=run_dir / "comparison_zone_html.png",
        heading_html_side_by_side_path=run_dir / "comparison_heading_html.png",
        contraction_html_side_by_side_path=run_dir / "comparison_contraction_html.png",
        llm_html_path=run_dir / "llm_html.png",
        zone_full_side_by_side_path=run_dir / "comparison_zone_full.png",
        heading_full_side_by_side_path=run_dir / "comparison_heading_full.png",
        contraction_full_side_by_side_path=run_dir / "comparison_contraction_full.png",
        llm_full_path=run_dir / "llm_full.png",
        zone_html_only_side_by_side_path=run_dir / "comparison_zone_html_only.png",
        heading_html_only_side_by_side_path=run_dir / "comparison_heading_html_only.png",
        contraction_html_only_side_by_side_path=run_dir / "comparison_contraction_html_only.png",
        llm_html_only_path=run_dir / "llm_html_only.png",
    )
    shutil.copyfile(run_dir / "comparison_zone.png", run_dir / "comparison.png")
    shutil.copyfile(run_dir / "zone.png", run_dir / "human.png")

def _save_analysis_with_clue(analyzer: DomTreeAnalyzer, result: AnalysisResult, run_dir: Path) -> None:
    record = result.to_dict()
    _write_json(run_dir / "result.json", record)
    (run_dir / "zone_tree.json").write_text(result.zone_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "heading_tree.json").write_text(result.heading_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "contraction_tree.json").write_text(result.contraction_tree.to_json(indent=2), encoding="utf-8")
    (run_dir / "llm_tree.json").write_text(result.llm_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_dom_tree:
        (run_dir / "llm_dom_tree.json").write_text(result.llm_dom_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_html_tree:
        (run_dir / "llm_html_tree.json").write_text(result.llm_html_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_full_tree:
        (run_dir / "llm_full_tree.json").write_text(result.llm_full_tree.to_json(indent=2), encoding="utf-8")
    if result.llm_html_only_tree:
        (run_dir / "llm_html_only_tree.json").write_text(result.llm_html_only_tree.to_json(indent=2), encoding="utf-8")

    _write_llm_comparison(result, run_dir)
    analyzer.visualize(
        result,
        zone_path=run_dir / "zone_clue.png",
        heading_path=run_dir / "heading_clue.png",
        contraction_path=run_dir / "contraction_clue.png",
        llm_path=run_dir / "llm_clue.png",
        zone_side_by_side_path=run_dir / "comparison_zone_clue.png",
        heading_side_by_side_path=run_dir / "comparison_heading_clue.png",
        contraction_side_by_side_path=run_dir / "comparison_contraction_clue.png",
        zone_dom_side_by_side_path=run_dir / "comparison_zone_dom_clue.png",
        heading_dom_side_by_side_path=run_dir / "comparison_heading_dom_clue.png",
        contraction_dom_side_by_side_path=run_dir / "comparison_contraction_dom_clue.png",
        llm_dom_path=run_dir / "llm_dom_clue.png",
        zone_html_side_by_side_path=run_dir / "comparison_zone_html_clue.png",
        heading_html_side_by_side_path=run_dir / "comparison_heading_html_clue.png",
        contraction_html_side_by_side_path=run_dir / "comparison_contraction_html_clue.png",
        llm_html_path=run_dir / "llm_html_clue.png",
        zone_full_side_by_side_path=run_dir / "comparison_zone_full_clue.png",
        heading_full_side_by_side_path=run_dir / "comparison_heading_full_clue.png",
        contraction_full_side_by_side_path=run_dir / "comparison_contraction_full_clue.png",
        llm_full_path=run_dir / "llm_full_clue.png",
        zone_html_only_side_by_side_path=run_dir / "comparison_zone_html_only_clue.png",
        heading_html_only_side_by_side_path=run_dir / "comparison_heading_html_only_clue.png",
        contraction_html_only_side_by_side_path=run_dir / "comparison_contraction_html_only_clue.png",
        llm_html_only_path=run_dir / "llm_html_only_clue.png",
        with_clues=True,
    )
    shutil.copyfile(run_dir / "comparison_zone_clue.png", run_dir / "comparison_clue.png")
    shutil.copyfile(run_dir / "zone_clue.png", run_dir / "human_clue.png")

def _save_batch_item(analyzer: DomTreeAnalyzer, result: AnalysisResult, batch_dir: Path, index: int) -> None:
    item_slug = _slugify(result.url)
    item_dir = batch_dir / f"{index:04d}_{item_slug}"
    item_dir.mkdir(parents=True, exist_ok=True)
    _save_analysis(analyzer, result, item_dir)

def _save_batch(results: Iterable[AnalysisResult], summary: dict, run_dir: Path) -> None:
    result_list = list(results)
    detailed_records = [result.to_dict() for result in result_list]
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "results.json", detailed_records)
    export_csv(result_list, run_dir / "results.csv")

def _write_llm_comparison(result: AnalysisResult, run_dir: Path) -> None:
    zone_vis = result.zone_comparison.metrics.flat()
    heading_vis = result.heading_comparison.metrics.flat()
    contraction_vis = result.contraction_comparison.metrics.flat()

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
        "contraction": {"vision": contraction_vis},
    }

    variants = [
        ("dom", result.zone_dom_comparison, result.heading_dom_comparison, result.contraction_dom_comparison),
        ("html", result.zone_html_comparison, result.heading_html_comparison, result.contraction_html_comparison),
        ("html_only", result.zone_html_only_comparison, result.heading_html_only_comparison, result.contraction_html_only_comparison),
        ("full", result.zone_full_comparison, result.heading_full_comparison, result.contraction_full_comparison),
    ]

    for key, zone_variant, heading_variant, contraction_variant in variants:
        if not (zone_variant and heading_variant and contraction_variant):
            continue
        zone_metrics = zone_variant.metrics.flat()
        heading_metrics = heading_variant.metrics.flat()
        contraction_metrics = contraction_variant.metrics.flat()

        comparison["zone"][f"vision_{key}"] = zone_metrics
        comparison["zone"][f"delta_{key}"] = _delta(zone_metrics, zone_vis)
        comparison["heading"][f"vision_{key}"] = heading_metrics
        comparison["heading"][f"delta_{key}"] = _delta(heading_metrics, heading_vis)
        comparison["contraction"][f"vision_{key}"] = contraction_metrics
        comparison["contraction"][f"delta_{key}"] = _delta(contraction_metrics, contraction_vis)

    if len(comparison["zone"]) == 1 and len(comparison["heading"]) == 1:
        return

    _write_json(run_dir / "llm_comparison.json", comparison)

def _save_component_metrics(result: AnalysisResult, run_dir: Path) -> None:
    def _metrics_for(tree: TreeNode | None) -> dict | None:
        if tree is None: return None
        clone = tree.copy()
        annotate_tree_with_categories(clone)
        return compute_category_stats(clone)

    metrics = {
        "zone": _metrics_for(result.zone_tree),
        "heading": _metrics_for(result.heading_tree),
        "contraction": _metrics_for(result.contraction_tree),
        "llm_vision": _metrics_for(result.llm_tree),
        "llm_dom": _metrics_for(result.llm_dom_tree),
        "llm_html": _metrics_for(result.llm_html_tree),
        "llm_html_only": _metrics_for(result.llm_html_only_tree),
        "llm_full": _metrics_for(result.llm_full_tree),
    }
    _write_json(run_dir / "component_metrics.json", metrics)

@app.command()
def analyze(url: str = typer.Argument(..., help="Target URL")) -> None:
    """Run the full pipeline for a single URL and persist outputs."""
    analyzer = _create_analyzer()
    result = analyzer.analyze_url(url)
    slug = _slugify(url)
    run_dir = _prepare_run_dir("single", slug)
    _save_analysis(analyzer, result, run_dir)

@app.command("analyze-with-metrics")
def analyze_with_metrics(url: str = typer.Argument(..., help="Target URL")) -> None:
    """Run analysis and also export category-level component metrics."""
    analyzer = _create_analyzer()
    result = analyzer.analyze_url(url)
    slug = _slugify(url)
    run_dir = _prepare_run_dir("single", slug)
    _save_analysis(analyzer, result, run_dir)
    _save_component_metrics(result, run_dir)

@app.command("analyze-with-clue")
def analyze_with_clue(url: str = typer.Argument(..., help="Target URL")) -> None:
    """Run analysis and also export category-level component metrics with clues."""
    analyzer = _create_analyzer()
    result = analyzer.analyze_url(url)
    slug = _slugify(url)
    run_dir = _prepare_run_dir("single", slug)
    _save_analysis_with_clue(analyzer, result, run_dir)
    _save_component_metrics(result, run_dir)

@app.command("analyze-offline")
def analyze_offline(
    html_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    screenshot_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    identifier: Optional[str] = typer.Argument(None, help="Optional identifier for saved outputs"),
) -> None:
    """Run analysis for pre-downloaded HTML and screenshot files."""
    analyzer = _create_analyzer()
    label = identifier or html_path.stem
    result = analyzer.analyze_offline(html_path=html_path, screenshot_path=screenshot_path, url=label)
    slug = _slugify(label)
    run_dir = _prepare_run_dir("offline", slug)
    _save_analysis(analyzer, result, run_dir)

@app.command("analyze-offline-with-metrics")
def analyze_offline_with_metrics(
    html_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    screenshot_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    identifier: Optional[str] = typer.Argument(None, help="Optional identifier for saved outputs"),
) -> None:
    """Offline variant that also exports category metrics."""
    analyzer = _create_analyzer()
    label = identifier or html_path.stem
    result = analyzer.analyze_offline(html_path=html_path, screenshot_path=screenshot_path, url=label)
    slug = _slugify(label)
    run_dir = _prepare_run_dir("offline", slug)
    _save_analysis(analyzer, result, run_dir)
    _save_component_metrics(result, run_dir)

@app.command()
def batch(
    batch_file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Text/CSV/JSON list of URLs"),
    identifier: Optional[str] = typer.Argument(None, help="Optional identifier for saved outputs"),
) -> None:
    """Run the pipeline on a batch of URLs and persist summary/records."""
    analyzer = _create_analyzer()
    label = identifier or batch_file.stem
    slug = _slugify(label)
    run_dir = _prepare_run_dir("batch", slug)
    items_dir = run_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    def _persist_result(result: AnalysisResult, index: int) -> None:
        _save_batch_item(analyzer, result, items_dir, index)

    results = run_batch_from_file(batch_file, analyzer, on_result=_persist_result)
    summary = analyzer.summarize(results)
    _save_batch(results, summary, run_dir)

if __name__ == "__main__":
    app()
