"""High-level pipeline orchestrating capture, extraction, and comparison."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from .capture import CaptureOptions, capture_page
from .comparison import ComparisonResult, compute_comparison
from .human_tree import HumanTreeBundle, HumanTreeExtractor, HumanTreeOptions
from .llm_tree import HeuristicLLMTreeGenerator, LLMTreeGenerator, LLMTreeRequest
from .visualization import plot_side_by_side, plot_tree
from .tree import TreeNode

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalysisResult:
    url: str
    screenshot_path: Path
    html_path: Path
    zone_tree: TreeNode
    heading_tree: TreeNode
    contraction_tree: TreeNode
    llm_tree: TreeNode
    zone_comparison: ComparisonResult
    heading_comparison: ComparisonResult
    contraction_comparison: ComparisonResult
    llm_dom_tree: TreeNode | None = None
    zone_dom_comparison: ComparisonResult | None = None
    heading_dom_comparison: ComparisonResult | None = None
    contraction_dom_comparison: ComparisonResult | None = None
    llm_html_tree: TreeNode | None = None
    zone_html_comparison: ComparisonResult | None = None
    heading_html_comparison: ComparisonResult | None = None
    contraction_html_comparison: ComparisonResult | None = None
    llm_html_only_tree: TreeNode | None = None
    zone_html_only_comparison: ComparisonResult | None = None
    heading_html_only_comparison: ComparisonResult | None = None
    contraction_html_only_comparison: ComparisonResult | None = None
    llm_full_tree: TreeNode | None = None
    zone_full_comparison: ComparisonResult | None = None
    heading_full_comparison: ComparisonResult | None = None
    contraction_full_comparison: ComparisonResult | None = None

    def to_dict(self) -> dict:
        payload = {
            "url": self.url,
            "screenshot_path": str(self.screenshot_path),
            "html_path": str(self.html_path),
            "metrics": {
                "zone": self.zone_comparison.metrics.flat(),
                "heading": self.heading_comparison.metrics.flat(),
                "contraction": self.contraction_comparison.metrics.flat(),
            },
            "mismatch_patterns": {
                "zone": self.zone_comparison.metrics.mismatch_patterns,
                "heading": self.heading_comparison.metrics.mismatch_patterns,
                "contraction": self.contraction_comparison.metrics.mismatch_patterns,
            },
        }
        if self.contraction_dom_comparison:
            payload["metrics"]["zone_dom"] = self.zone_dom_comparison.metrics.flat()
            payload["metrics"]["heading_dom"] = self.heading_dom_comparison.metrics.flat()
            payload["metrics"]["contraction_dom"] = self.contraction_dom_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_dom"] = self.zone_dom_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_dom"] = self.heading_dom_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["contraction_dom"] = self.contraction_dom_comparison.metrics.mismatch_patterns
        
        if self.contraction_html_comparison:
            payload["metrics"]["zone_html"] = self.zone_html_comparison.metrics.flat()
            payload["metrics"]["heading_html"] = self.heading_html_comparison.metrics.flat()
            payload["metrics"]["contraction_html"] = self.contraction_html_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_html"] = self.zone_html_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_html"] = self.heading_html_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["contraction_html"] = self.contraction_html_comparison.metrics.mismatch_patterns

        if self.contraction_html_only_comparison:
            payload["metrics"]["zone_html_only"] = self.zone_html_only_comparison.metrics.flat()
            payload["metrics"]["heading_html_only"] = self.heading_html_only_comparison.metrics.flat()
            payload["metrics"]["contraction_html_only"] = self.contraction_html_only_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_html_only"] = self.zone_html_only_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_html_only"] = self.heading_html_only_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["contraction_html_only"] = self.contraction_html_only_comparison.metrics.mismatch_patterns

        if self.contraction_full_comparison:
            payload["metrics"]["zone_full"] = self.zone_full_comparison.metrics.flat()
            payload["metrics"]["heading_full"] = self.heading_full_comparison.metrics.flat()
            payload["metrics"]["contraction_full"] = self.contraction_full_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_full"] = self.zone_full_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_full"] = self.heading_full_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["contraction_full"] = self.contraction_full_comparison.metrics.mismatch_patterns
            
        return payload


class DomTreeAnalyzer:
    def __init__(
        self,
        *,
        capture_options: Optional[CaptureOptions] = None,
        human_options: Optional[HumanTreeOptions] = None,
        llm_generator: Optional[LLMTreeGenerator] = None,
        dom_llm_generator: Optional[LLMTreeGenerator] = None,
        html_llm_generator: Optional[LLMTreeGenerator] = None,
        html_only_llm_generator: Optional[LLMTreeGenerator] = None,
        full_llm_generator: Optional[LLMTreeGenerator] = None,
    ):
        self.capture_options = capture_options or CaptureOptions()
        self.human_options = human_options or HumanTreeOptions()
        self.llm_generator = llm_generator or HeuristicLLMTreeGenerator()
        self.dom_llm_generator = dom_llm_generator
        self.html_llm_generator = html_llm_generator
        self.html_only_llm_generator = html_only_llm_generator
        self.full_llm_generator = full_llm_generator

    def analyze_url(self, url: str, *, name: Optional[str] = None) -> AnalysisResult:
        capture = capture_page(url, options=self.capture_options, name=name)
        html = Path(capture["html_path"]).read_text(encoding="utf-8")
        human_tree_bundle = HumanTreeExtractor(html, url=url, options=self.human_options).extract()

        llm_tree = self.llm_generator.generate(
            LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
        )
        zone_comparison = compute_comparison(human_tree_bundle.zone_tree, llm_tree)
        heading_comparison = compute_comparison(human_tree_bundle.heading_tree, llm_tree)
        contraction_comparison = compute_comparison(human_tree_bundle.contraction_tree, llm_tree)

        llm_dom_tree: Optional[TreeNode] = None
        zone_dom_comparison: Optional[ComparisonResult] = None
        heading_dom_comparison: Optional[ComparisonResult] = None
        contraction_dom_comparison: Optional[ComparisonResult] = None
        if self.dom_llm_generator is not None:
            dom_tree = self.dom_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_dom_tree = dom_tree
            zone_dom_comparison = compute_comparison(human_tree_bundle.zone_tree, dom_tree)
            heading_dom_comparison = compute_comparison(human_tree_bundle.heading_tree, dom_tree)
            contraction_dom_comparison = compute_comparison(human_tree_bundle.contraction_tree, dom_tree)

        llm_html_tree: Optional[TreeNode] = None
        zone_html_comparison: Optional[ComparisonResult] = None
        heading_html_comparison: Optional[ComparisonResult] = None
        contraction_html_comparison: Optional[ComparisonResult] = None
        if self.html_llm_generator is not None:
            html_tree = self.html_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_html_tree = html_tree
            zone_html_comparison = compute_comparison(human_tree_bundle.zone_tree, html_tree)
            heading_html_comparison = compute_comparison(human_tree_bundle.heading_tree, html_tree)
            contraction_html_comparison = compute_comparison(human_tree_bundle.contraction_tree, html_tree)

        llm_html_only_tree: Optional[TreeNode] = None
        zone_html_only_comparison: Optional[ComparisonResult] = None
        heading_html_only_comparison: Optional[ComparisonResult] = None
        contraction_html_only_comparison: Optional[ComparisonResult] = None
        if self.html_only_llm_generator is not None:
            html_only_tree = self.html_only_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_html_only_tree = html_only_tree
            zone_html_only_comparison = compute_comparison(human_tree_bundle.zone_tree, html_only_tree)
            heading_html_only_comparison = compute_comparison(human_tree_bundle.heading_tree, html_only_tree)
            contraction_html_only_comparison = compute_comparison(human_tree_bundle.contraction_tree, html_only_tree)

        llm_full_tree: Optional[TreeNode] = None
        zone_full_comparison: Optional[ComparisonResult] = None
        heading_full_comparison: Optional[ComparisonResult] = None
        contraction_full_comparison: Optional[ComparisonResult] = None
        if self.full_llm_generator is not None:
            full_tree = self.full_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_full_tree = full_tree
            zone_full_comparison = compute_comparison(human_tree_bundle.zone_tree, full_tree)
            heading_full_comparison = compute_comparison(human_tree_bundle.heading_tree, full_tree)
            contraction_full_comparison = compute_comparison(human_tree_bundle.contraction_tree, full_tree)

        return AnalysisResult(
            url=url,
            screenshot_path=Path(capture["screenshot_path"]),
            html_path=Path(capture["html_path"]),
            zone_tree=human_tree_bundle.zone_tree,
            heading_tree=human_tree_bundle.heading_tree,
            contraction_tree=human_tree_bundle.contraction_tree,
            llm_tree=llm_tree,
            zone_comparison=zone_comparison,
            heading_comparison=heading_comparison,
            contraction_comparison=contraction_comparison,
            llm_dom_tree=llm_dom_tree,
            zone_dom_comparison=zone_dom_comparison,
            heading_dom_comparison=heading_dom_comparison,
            contraction_dom_comparison=contraction_dom_comparison,
            llm_html_tree=llm_html_tree,
            zone_html_comparison=zone_html_comparison,
            heading_html_comparison=heading_html_comparison,
            contraction_html_comparison=contraction_html_comparison,
            llm_html_only_tree=llm_html_only_tree,
            zone_html_only_comparison=zone_html_only_comparison,
            heading_html_only_comparison=heading_html_only_comparison,
            contraction_html_only_comparison=contraction_html_only_comparison,
            llm_full_tree=llm_full_tree,
            zone_full_comparison=zone_full_comparison,
            heading_full_comparison=heading_full_comparison,
            contraction_full_comparison=contraction_full_comparison,
        )

    def analyze_offline(self, *, html_path: Path, screenshot_path: Path, url: str = "offline") -> AnalysisResult:
        html = html_path.read_text(encoding="utf-8")
        human_tree_bundle = HumanTreeExtractor(html, url=url, options=self.human_options).extract()

        llm_tree = self.llm_generator.generate(LLMTreeRequest(screenshot_path=screenshot_path, html=html))
        zone_comparison = compute_comparison(human_tree_bundle.zone_tree, llm_tree)
        heading_comparison = compute_comparison(human_tree_bundle.heading_tree, llm_tree)
        contraction_comparison = compute_comparison(human_tree_bundle.contraction_tree, llm_tree)

        llm_dom_tree: Optional[TreeNode] = None
        zone_dom_comparison: Optional[ComparisonResult] = None
        heading_dom_comparison: Optional[ComparisonResult] = None
        contraction_dom_comparison: Optional[ComparisonResult] = None
        if self.dom_llm_generator is not None:
            dom_tree = self.dom_llm_generator.generate(LLMTreeRequest(screenshot_path=screenshot_path, html=html))
            llm_dom_tree = dom_tree
            zone_dom_comparison = compute_comparison(human_tree_bundle.zone_tree, dom_tree)
            heading_dom_comparison = compute_comparison(human_tree_bundle.heading_tree, dom_tree)
            contraction_dom_comparison = compute_comparison(human_tree_bundle.contraction_tree, dom_tree)

        # ... (repeat for all llm variants)

        return AnalysisResult(
            url=url,
            screenshot_path=screenshot_path,
            html_path=html_path,
            zone_tree=human_tree_bundle.zone_tree,
            heading_tree=human_tree_bundle.heading_tree,
            contraction_tree=human_tree_bundle.contraction_tree,
            llm_tree=llm_tree,
            zone_comparison=zone_comparison,
            heading_comparison=heading_comparison,
            contraction_comparison=contraction_comparison,
            llm_dom_tree=llm_dom_tree,
            zone_dom_comparison=zone_dom_comparison,
            heading_dom_comparison=heading_dom_comparison,
            contraction_dom_comparison=contraction_dom_comparison,
            # ... (and all other variants)
        )

    def run_batch(
        self,
        urls: Sequence[str],
        *,
        on_result: Callable[[AnalysisResult, int], None] | None = None,
    ) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        for index, url in enumerate(urls, start=1):
            try:
                logger.info("[%s/%s] Processing %s", index, len(urls), url)
                result = self.analyze_url(url)
                results.append(result)
                if on_result:
                    on_result(result, index)
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Failed to process %s: %s", url, exc)
        return results

    def summarize(self, analyses: Iterable[AnalysisResult]) -> dict:
        analyses = list(analyses)
        # Initialize lists for metrics
        zone_metrics, heading_metrics, contraction_metrics = [], [], []
        zone_dom_metrics, heading_dom_metrics, contraction_dom_metrics = [], [], []
        zone_html_metrics, heading_html_metrics, contraction_html_metrics = [], [], []
        zone_html_only_metrics, heading_html_only_metrics, contraction_html_only_metrics = [], [], []
        zone_full_metrics, heading_full_metrics, contraction_full_metrics = [], [], []

        # Initialize dicts for mismatch totals
        mismatch_keys = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}
        zone_mismatch, heading_mismatch, contraction_mismatch = mismatch_keys.copy(), mismatch_keys.copy(), mismatch_keys.copy()
        zone_dom_mismatch, heading_dom_mismatch, contraction_dom_mismatch = mismatch_keys.copy(), mismatch_keys.copy(), mismatch_keys.copy()
        zone_html_mismatch, heading_html_mismatch, contraction_html_mismatch = mismatch_keys.copy(), mismatch_keys.copy(), mismatch_keys.copy()
        zone_html_only_mismatch, heading_html_only_mismatch, contraction_html_only_mismatch = mismatch_keys.copy(), mismatch_keys.copy(), mismatch_keys.copy()
        zone_full_mismatch, heading_full_mismatch, contraction_full_mismatch = mismatch_keys.copy(), mismatch_keys.copy(), mismatch_keys.copy()

        def _accumulate(totals: dict, patterns: dict):
            totals["missing"] += patterns.get("missing_nodes", {}).get("count", 0)
            totals["extra"] += patterns.get("extra_nodes", {}).get("count", 0)
            totals["depth_shift"] += patterns.get("depth_shift", {}).get("count", 0)
            totals["order"] += patterns.get("reading_order", {}).get("gaps", 0)

        for analysis in analyses:
            zone_metrics.append(analysis.zone_comparison.metrics.flat())
            heading_metrics.append(analysis.heading_comparison.metrics.flat())
            contraction_metrics.append(analysis.contraction_comparison.metrics.flat())
            _accumulate(zone_mismatch, analysis.zone_comparison.metrics.mismatch_patterns)
            _accumulate(heading_mismatch, analysis.heading_comparison.metrics.mismatch_patterns)
            _accumulate(contraction_mismatch, analysis.contraction_comparison.metrics.mismatch_patterns)

            if analysis.contraction_dom_comparison:
                zone_dom_metrics.append(analysis.zone_dom_comparison.metrics.flat())
                heading_dom_metrics.append(analysis.heading_dom_comparison.metrics.flat())
                contraction_dom_metrics.append(analysis.contraction_dom_comparison.metrics.flat())
                _accumulate(zone_dom_mismatch, analysis.zone_dom_comparison.metrics.mismatch_patterns)
                _accumulate(heading_dom_mismatch, analysis.heading_dom_comparison.metrics.mismatch_patterns)
                _accumulate(contraction_dom_mismatch, analysis.contraction_dom_comparison.metrics.mismatch_patterns)

            if analysis.contraction_html_comparison:
                zone_html_metrics.append(analysis.zone_html_comparison.metrics.flat())
                heading_html_metrics.append(analysis.heading_html_comparison.metrics.flat())
                contraction_html_metrics.append(analysis.contraction_html_comparison.metrics.flat())
                _accumulate(zone_html_mismatch, analysis.zone_html_comparison.metrics.mismatch_patterns)
                _accumulate(heading_html_mismatch, analysis.heading_html_comparison.metrics.mismatch_patterns)
                _accumulate(contraction_html_mismatch, analysis.contraction_html_comparison.metrics.mismatch_patterns)

            if analysis.contraction_html_only_comparison:
                zone_html_only_metrics.append(analysis.zone_html_only_comparison.metrics.flat())
                heading_html_only_metrics.append(analysis.heading_html_only_comparison.metrics.flat())
                contraction_html_only_metrics.append(analysis.contraction_html_only_comparison.metrics.flat())
                _accumulate(zone_html_only_mismatch, analysis.zone_html_only_comparison.metrics.mismatch_patterns)
                _accumulate(heading_html_only_mismatch, analysis.heading_html_only_comparison.metrics.mismatch_patterns)
                _accumulate(contraction_html_only_mismatch, analysis.contraction_html_only_comparison.metrics.mismatch_patterns)

            if analysis.contraction_full_comparison:
                zone_full_metrics.append(analysis.zone_full_comparison.metrics.flat())
                heading_full_metrics.append(analysis.heading_full_comparison.metrics.flat())
                contraction_full_metrics.append(analysis.contraction_full_comparison.metrics.flat())
                _accumulate(zone_full_mismatch, analysis.zone_full_comparison.metrics.mismatch_patterns)
                _accumulate(heading_full_mismatch, analysis.heading_full_comparison.metrics.mismatch_patterns)
                _accumulate(contraction_full_mismatch, analysis.contraction_full_comparison.metrics.mismatch_patterns)

        def _average(metrics_list: List[dict]) -> dict:
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            return {key: sum(m.get(key, 0) for m in metrics_list) / len(metrics_list) for key in keys}

        def _build_summary(metrics: list, mismatches: dict) -> dict:
            if not metrics:
                return {}
            return {
                "average_metrics": _average(metrics),
                "mismatch_totals": mismatches,
                "count": len(metrics),
            }

        summary = {
            "zone": _build_summary(zone_metrics, zone_mismatch),
            "heading": _build_summary(heading_metrics, heading_mismatch),
            "contraction": _build_summary(contraction_metrics, contraction_mismatch),
        }

        for key, metrics, mismatches in [
            ("zone_dom", zone_dom_metrics, zone_dom_mismatch),
            ("heading_dom", heading_dom_metrics, heading_dom_mismatch),
            ("contraction_dom", contraction_dom_metrics, contraction_dom_mismatch),
            ("zone_html", zone_html_metrics, zone_html_mismatch),
            ("heading_html", heading_html_metrics, heading_html_mismatch),
            ("contraction_html", contraction_html_metrics, contraction_html_mismatch),
            ("zone_html_only", zone_html_only_metrics, zone_html_only_mismatch),
            ("heading_html_only", heading_html_only_metrics, heading_html_only_mismatch),
            ("contraction_html_only", contraction_html_only_metrics, contraction_html_only_mismatch),
            ("zone_full", zone_full_metrics, zone_full_mismatch),
            ("heading_full", heading_full_metrics, heading_full_mismatch),
            ("contraction_full", contraction_full_metrics, contraction_full_mismatch),
        ]:
            if metrics:
                summary[key] = _build_summary(metrics, mismatches)

        return summary

    def visualize(
        self,
        analysis: AnalysisResult,
        *,
        zone_side_by_side_path: Optional[Path] = None,
        heading_side_by_side_path: Optional[Path] = None,
        contraction_side_by_side_path: Optional[Path] = None,
        zone_path: Optional[Path] = None,
        heading_path: Optional[Path] = None,
        contraction_path: Optional[Path] = None,
        llm_path: Optional[Path] = None,
        zone_dom_side_by_side_path: Optional[Path] = None,
        heading_dom_side_by_side_path: Optional[Path] = None,
        contraction_dom_side_by_side_path: Optional[Path] = None,
        llm_dom_path: Optional[Path] = None,
        zone_html_side_by_side_path: Optional[Path] = None,
        heading_html_side_by_side_path: Optional[Path] = None,
        contraction_html_side_by_side_path: Optional[Path] = None,
        llm_html_path: Optional[Path] = None,
        zone_full_side_by_side_path: Optional[Path] = None,
        heading_full_side_by_side_path: Optional[Path] = None,
        contraction_full_side_by_side_path: Optional[Path] = None,
        llm_full_path: Optional[Path] = None,
        zone_html_only_side_by_side_path: Optional[Path] = None,
        heading_html_only_side_by_side_path: Optional[Path] = None,
        contraction_html_only_side_by_side_path: Optional[Path] = None,
        llm_html_only_path: Optional[Path] = None,
        with_clues: bool = False,
    ) -> None:
        if zone_path:
            plot_tree(analysis.zone_tree, title="Zone Tree", path=zone_path, with_clues=with_clues)
        if heading_path:
            plot_tree(analysis.heading_tree, title="Heading Tree", path=heading_path, with_clues=with_clues)
        if contraction_path:
            plot_tree(analysis.contraction_tree, title="Contraction Tree", path=contraction_path, with_clues=with_clues)
        if llm_path:
            plot_tree(analysis.llm_tree, title="LLM Tree (Vision)", path=llm_path, with_clues=with_clues)

        if zone_side_by_side_path:
            plot_side_by_side(analysis.zone_tree, analysis.llm_tree, path=zone_side_by_side_path, with_clues=with_clues)
        if heading_side_by_side_path:
            plot_side_by_side(analysis.heading_tree, analysis.llm_tree, path=heading_side_by_side_path, with_clues=with_clues)
        if contraction_side_by_side_path:
            plot_side_by_side(analysis.contraction_tree, analysis.llm_tree, path=contraction_side_by_side_path, with_clues=with_clues)

        if analysis.llm_dom_tree:
            if zone_dom_side_by_side_path:
                plot_side_by_side(analysis.zone_tree, analysis.llm_dom_tree, path=zone_dom_side_by_side_path, with_clues=with_clues)
            if heading_dom_side_by_side_path:
                plot_side_by_side(analysis.heading_tree, analysis.llm_dom_tree, path=heading_dom_side_by_side_path, with_clues=with_clues)
            if contraction_dom_side_by_side_path:
                plot_side_by_side(analysis.contraction_tree, analysis.llm_dom_tree, path=contraction_dom_side_by_side_path, with_clues=with_clues)
            if llm_dom_path:
                plot_tree(analysis.llm_dom_tree, title="LLM Tree (Vision + DOM)", path=llm_dom_path, with_clues=with_clues)

        if analysis.llm_html_tree:
            if zone_html_side_by_side_path:
                plot_side_by_side(analysis.zone_tree, analysis.llm_html_tree, path=zone_html_side_by_side_path, with_clues=with_clues)
            if heading_html_side_by_side_path:
                plot_side_by_side(analysis.heading_tree, analysis.llm_html_tree, path=heading_html_side_by_side_path, with_clues=with_clues)
            if contraction_html_side_by_side_path:
                plot_side_by_side(analysis.contraction_tree, analysis.llm_html_tree, path=contraction_html_side_by_side_path, with_clues=with_clues)
            if llm_html_path:
                plot_tree(analysis.llm_html_tree, title="LLM Tree (Vision + HTML)", path=llm_html_path, with_clues=with_clues)

        if analysis.llm_html_only_tree:
            if zone_html_only_side_by_side_path:
                plot_side_by_side(analysis.zone_tree, analysis.llm_html_only_tree, path=zone_html_only_side_by_side_path, with_clues=with_clues)
            if heading_html_only_side_by_side_path:
                plot_side_by_side(analysis.heading_tree, analysis.llm_html_only_tree, path=heading_html_only_side_by_side_path, with_clues=with_clues)
            if contraction_html_only_side_by_side_path:
                plot_side_by_side(analysis.contraction_tree, analysis.llm_html_only_tree, path=contraction_html_only_side_by_side_path, with_clues=with_clues)
            if llm_html_only_path:
                plot_tree(analysis.llm_html_only_tree, title="LLM Tree (HTML Only)", path=llm_html_only_path, with_clues=with_clues)

        if analysis.llm_full_tree:
            if zone_full_side_by_side_path:
                plot_side_by_side(analysis.zone_tree, analysis.llm_full_tree, path=zone_full_side_by_side_path, with_clues=with_clues)
            if heading_full_side_by_side_path:
                plot_side_by_side(analysis.heading_tree, analysis.llm_full_tree, path=heading_full_side_by_side_path, with_clues=with_clues)
            if contraction_full_side_by_side_path:
                plot_side_by_side(analysis.contraction_tree, analysis.llm_full_tree, path=contraction_full_side_by_side_path, with_clues=with_clues)
            if llm_full_path:
                plot_tree(analysis.llm_full_tree, title="LLM Tree (Vision + DOM + HTML)", path=llm_full_path, with_clues=with_clues)
