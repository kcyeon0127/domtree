"""High-level pipeline orchestrating capture, extraction, and comparison."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .capture import CaptureOptions, capture_page
from .comparison import ComparisonResult, compute_comparison
from .human_tree import HumanTreeExtractor, HumanTreeOptions
from .llm_tree import HeuristicLLMTreeGenerator, LLMTreeGenerator, LLMTreeRequest
from .visualization import plot_side_by_side, plot_tree
from .tree import TreeNode

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalysisResult:
    url: str
    screenshot_path: Path
    html_path: Path
    human_tree: TreeNode
    llm_tree: TreeNode
    comparison: ComparisonResult

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "screenshot_path": str(self.screenshot_path),
            "html_path": str(self.html_path),
            "metrics": self.comparison.metrics.flat(),
            "mismatch_patterns": self.comparison.metrics.mismatch_patterns,
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
        human_tree = HumanTreeExtractor(html, url=url, options=self.human_options).extract()
        llm_tree = self.llm_generator.generate(LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html))
        comparison = compute_comparison(human_tree, llm_tree)
        return AnalysisResult(
            url=url,
            screenshot_path=Path(capture["screenshot_path"]),
            html_path=Path(capture["html_path"]),
            human_tree=human_tree,
            llm_tree=llm_tree,
            comparison=comparison,
        )

    def analyze_offline(self, *, html_path: Path, screenshot_path: Path, url: str = "offline") -> AnalysisResult:
        html = html_path.read_text(encoding="utf-8")
        human_tree = HumanTreeExtractor(html, url=url, options=self.human_options).extract()
        llm_tree = self.llm_generator.generate(LLMTreeRequest(screenshot_path=screenshot_path, html=html))
        comparison = compute_comparison(human_tree, llm_tree)
        return AnalysisResult(
            url=url,
            screenshot_path=screenshot_path,
            html_path=html_path,
            human_tree=human_tree,
            llm_tree=llm_tree,
            comparison=comparison,
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
        metrics = []
        mismatch_counts = {
            "missing": 0,
            "extra": 0,
            "depth_shift": 0,
            "order": 0,
        }
        for analysis in analyses:
            flat = analysis.comparison.metrics.flat()
            metrics.append(flat)
            mismatch = analysis.comparison.metrics.mismatch_patterns
            mismatch_counts["missing"] += mismatch["missing_nodes"]["count"]
            mismatch_counts["extra"] += mismatch["extra_nodes"]["count"]
            mismatch_counts["depth_shift"] += mismatch["depth_shift"]["count"]
            mismatch_counts["order"] += mismatch["reading_order"]["gaps"]
        average_metrics = {}
        if metrics:
            keys = metrics[0].keys()
            for key in keys:
                average_metrics[key] = sum(m[key] for m in metrics) / len(metrics)
        return {
            "average_metrics": average_metrics,
            "mismatch_totals": mismatch_counts,
            "count": len(metrics),
        }

    def visualize(
        self,
        analysis: AnalysisResult,
        *,
        side_by_side_path: Optional[Path] = None,
        human_path: Optional[Path] = None,
        llm_path: Optional[Path] = None,
    ) -> None:
        if side_by_side_path:
            plot_side_by_side(analysis.human_tree, analysis.llm_tree, path=side_by_side_path)
        if human_path:
            plot_tree(analysis.human_tree, title="Human Tree", path=human_path)
        if llm_path:
            plot_tree(analysis.llm_tree, title="LLM Tree", path=llm_path)
