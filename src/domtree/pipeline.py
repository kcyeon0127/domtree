"""High-level pipeline orchestrating capture, extraction, and comparison."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

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
    human_zone_tree: TreeNode
    human_heading_tree: TreeNode
    llm_tree: TreeNode
    zone_comparison: ComparisonResult
    heading_comparison: ComparisonResult

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "screenshot_path": str(self.screenshot_path),
            "html_path": str(self.html_path),
            "metrics": {
                "zone": self.zone_comparison.metrics.flat(),
                "heading": self.heading_comparison.metrics.flat(),
            },
            "mismatch_patterns": {
                "zone": self.zone_comparison.metrics.mismatch_patterns,
                "heading": self.heading_comparison.metrics.mismatch_patterns,
            },
        }


class DomTreeAnalyzer:
    def __init__(
        self,
        *,
        capture_options: Optional[CaptureOptions] = None,
        human_options: Optional[HumanTreeOptions] = None,
        llm_generator: Optional[LLMTreeGenerator] = None,
    ):
        self.capture_options = capture_options or CaptureOptions()
        self.human_options = human_options or HumanTreeOptions()
        self.llm_generator = llm_generator or HeuristicLLMTreeGenerator()

    def analyze_url(self, url: str, *, name: Optional[str] = None) -> AnalysisResult:
        capture = capture_page(url, options=self.capture_options, name=name)
        html = Path(capture["html_path"]).read_text(encoding="utf-8")
        human_trees = HumanTreeExtractor(html, url=url, options=self.human_options).extract()
        llm_tree = self.llm_generator.generate(LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html))
        zone_comparison = compute_comparison(human_trees.zone_tree, llm_tree)
        heading_comparison = compute_comparison(human_trees.heading_tree, llm_tree)
        return AnalysisResult(
            url=url,
            screenshot_path=Path(capture["screenshot_path"]),
            html_path=Path(capture["html_path"]),
            human_zone_tree=human_trees.zone_tree,
            human_heading_tree=human_trees.heading_tree,
            llm_tree=llm_tree,
            zone_comparison=zone_comparison,
            heading_comparison=heading_comparison,
        )

    def analyze_offline(self, *, html_path: Path, screenshot_path: Path, url: str = "offline") -> AnalysisResult:
        html = html_path.read_text(encoding="utf-8")
        human_trees = HumanTreeExtractor(html, url=url, options=self.human_options).extract()
        llm_tree = self.llm_generator.generate(LLMTreeRequest(screenshot_path=screenshot_path, html=html))
        zone_comparison = compute_comparison(human_trees.zone_tree, llm_tree)
        heading_comparison = compute_comparison(human_trees.heading_tree, llm_tree)
        return AnalysisResult(
            url=url,
            screenshot_path=screenshot_path,
            html_path=html_path,
            human_zone_tree=human_trees.zone_tree,
            human_heading_tree=human_trees.heading_tree,
            llm_tree=llm_tree,
            zone_comparison=zone_comparison,
            heading_comparison=heading_comparison,
        )

    def run_batch(self, urls: Sequence[str]) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        for index, url in enumerate(urls, start=1):
            try:
                logger.info("[%s/%s] Processing %s", index, len(urls), url)
                result = self.analyze_url(url)
                results.append(result)
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Failed to process %s: %s", url, exc)
        return results

    def summarize(self, analyses: Iterable[AnalysisResult]) -> dict:
        zone_metrics: List[dict] = []
        heading_metrics: List[dict] = []
        zone_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}
        heading_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}

        for analysis in analyses:
            zflat = analysis.zone_comparison.metrics.flat()
            hflat = analysis.heading_comparison.metrics.flat()
            zone_metrics.append(zflat)
            heading_metrics.append(hflat)

            zm = analysis.zone_comparison.metrics.mismatch_patterns
            hm = analysis.heading_comparison.metrics.mismatch_patterns
            zone_mismatch["missing"] += zm["missing_nodes"]["count"]
            zone_mismatch["extra"] += zm["extra_nodes"]["count"]
            zone_mismatch["depth_shift"] += zm["depth_shift"]["count"]
            zone_mismatch["order"] += zm["reading_order"]["gaps"]

            heading_mismatch["missing"] += hm["missing_nodes"]["count"]
            heading_mismatch["extra"] += hm["extra_nodes"]["count"]
            heading_mismatch["depth_shift"] += hm["depth_shift"]["count"]
            heading_mismatch["order"] += hm["reading_order"]["gaps"]

        def _average(metrics_list: List[dict]) -> dict:
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            return {
                key: sum(m[key] for m in metrics_list) / len(metrics_list)
                for key in keys
            }

        return {
            "zone": {
                "average_metrics": _average(zone_metrics),
                "mismatch_totals": zone_mismatch,
                "count": len(zone_metrics),
            },
            "heading": {
                "average_metrics": _average(heading_metrics),
                "mismatch_totals": heading_mismatch,
                "count": len(heading_metrics),
            },
        }

    def visualize(
        self,
        analysis: AnalysisResult,
        *,
        zone_side_by_side_path: Optional[Path] = None,
        heading_side_by_side_path: Optional[Path] = None,
        zone_path: Optional[Path] = None,
        heading_path: Optional[Path] = None,
        llm_path: Optional[Path] = None,
    ) -> None:
        if zone_side_by_side_path:
            plot_side_by_side(analysis.zone_comparison.human_tree, analysis.llm_tree, path=zone_side_by_side_path)
        if heading_side_by_side_path:
            plot_side_by_side(analysis.heading_comparison.human_tree, analysis.llm_tree, path=heading_side_by_side_path)
        if zone_path:
            plot_tree(analysis.human_zone_tree, title="Human Zone Tree", path=zone_path)
        if heading_path:
            plot_tree(analysis.human_heading_tree, title="Human Heading Tree", path=heading_path)
        if llm_path:
            plot_tree(analysis.llm_tree, title="LLM Tree", path=llm_path)
