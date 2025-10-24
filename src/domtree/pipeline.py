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
    llm_dom_tree: TreeNode | None = None
    zone_dom_comparison: ComparisonResult | None = None
    heading_dom_comparison: ComparisonResult | None = None
    llm_html_tree: TreeNode | None = None
    zone_html_comparison: ComparisonResult | None = None
    heading_html_comparison: ComparisonResult | None = None
    llm_html_only_tree: TreeNode | None = None
    zone_html_only_comparison: ComparisonResult | None = None
    heading_html_only_comparison: ComparisonResult | None = None
    llm_full_tree: TreeNode | None = None
    zone_full_comparison: ComparisonResult | None = None
    heading_full_comparison: ComparisonResult | None = None

    def to_dict(self) -> dict:
        payload = {
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
        if self.zone_dom_comparison and self.heading_dom_comparison:
            payload["metrics"]["zone_dom"] = self.zone_dom_comparison.metrics.flat()
            payload["metrics"]["heading_dom"] = self.heading_dom_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_dom"] = self.zone_dom_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_dom"] = self.heading_dom_comparison.metrics.mismatch_patterns
        if self.zone_html_comparison and self.heading_html_comparison:
            payload["metrics"]["zone_html"] = self.zone_html_comparison.metrics.flat()
            payload["metrics"]["heading_html"] = self.heading_html_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_html"] = self.zone_html_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_html"] = self.heading_html_comparison.metrics.mismatch_patterns
        if self.zone_html_only_comparison and self.heading_html_only_comparison:
            payload["metrics"]["zone_html_only"] = self.zone_html_only_comparison.metrics.flat()
            payload["metrics"]["heading_html_only"] = self.heading_html_only_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_html_only"] = self.zone_html_only_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_html_only"] = self.heading_html_only_comparison.metrics.mismatch_patterns
        if self.zone_full_comparison and self.heading_full_comparison:
            payload["metrics"]["zone_full"] = self.zone_full_comparison.metrics.flat()
            payload["metrics"]["heading_full"] = self.heading_full_comparison.metrics.flat()
            payload["mismatch_patterns"]["zone_full"] = self.zone_full_comparison.metrics.mismatch_patterns
            payload["mismatch_patterns"]["heading_full"] = self.heading_full_comparison.metrics.mismatch_patterns
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
        human_trees = HumanTreeExtractor(html, url=url, options=self.human_options).extract()
        llm_tree = self.llm_generator.generate(
            LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
        )
        zone_comparison = compute_comparison(human_trees.zone_tree, llm_tree)
        heading_comparison = compute_comparison(human_trees.heading_tree, llm_tree)

        llm_dom_tree = None
        zone_dom_comparison = None
        heading_dom_comparison = None
        if self.dom_llm_generator is not None:
            dom_tree = self.dom_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_dom_tree = dom_tree
            zone_dom_comparison = compute_comparison(human_trees.zone_tree, dom_tree)
            heading_dom_comparison = compute_comparison(human_trees.heading_tree, dom_tree)
        llm_html_tree = None
        zone_html_comparison = None
        heading_html_comparison = None
        if self.html_llm_generator is not None:
            html_tree = self.html_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_html_tree = html_tree
            zone_html_comparison = compute_comparison(human_trees.zone_tree, html_tree)
            heading_html_comparison = compute_comparison(human_trees.heading_tree, html_tree)
        llm_html_only_tree = None
        zone_html_only_comparison = None
        heading_html_only_comparison = None
        if self.html_only_llm_generator is not None:
            html_only_tree = self.html_only_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_html_only_tree = html_only_tree
            zone_html_only_comparison = compute_comparison(human_trees.zone_tree, html_only_tree)
            heading_html_only_comparison = compute_comparison(human_trees.heading_tree, html_only_tree)
        llm_full_tree = None
        zone_full_comparison = None
        heading_full_comparison = None
        if self.full_llm_generator is not None:
            full_tree = self.full_llm_generator.generate(
                LLMTreeRequest(screenshot_path=Path(capture["screenshot_path"]), html=html)
            )
            llm_full_tree = full_tree
            zone_full_comparison = compute_comparison(human_trees.zone_tree, full_tree)
            heading_full_comparison = compute_comparison(human_trees.heading_tree, full_tree)
        return AnalysisResult(
            url=url,
            screenshot_path=Path(capture["screenshot_path"]),
            html_path=Path(capture["html_path"]),
            human_zone_tree=human_trees.zone_tree,
            human_heading_tree=human_trees.heading_tree,
            llm_tree=llm_tree,
            zone_comparison=zone_comparison,
            heading_comparison=heading_comparison,
            llm_dom_tree=llm_dom_tree,
            zone_dom_comparison=zone_dom_comparison,
            heading_dom_comparison=heading_dom_comparison,
            llm_html_tree=llm_html_tree,
            zone_html_comparison=zone_html_comparison,
            heading_html_comparison=heading_html_comparison,
            llm_html_only_tree=llm_html_only_tree,
            zone_html_only_comparison=zone_html_only_comparison,
            heading_html_only_comparison=heading_html_only_comparison,
            llm_full_tree=llm_full_tree,
            zone_full_comparison=zone_full_comparison,
            heading_full_comparison=heading_full_comparison,
        )

    def analyze_offline(self, *, html_path: Path, screenshot_path: Path, url: str = "offline") -> AnalysisResult:
        html = html_path.read_text(encoding="utf-8")
        human_trees = HumanTreeExtractor(html, url=url, options=self.human_options).extract()
        llm_tree = self.llm_generator.generate(
            LLMTreeRequest(screenshot_path=screenshot_path, html=html)
        )
        zone_comparison = compute_comparison(human_trees.zone_tree, llm_tree)
        heading_comparison = compute_comparison(human_trees.heading_tree, llm_tree)

        llm_dom_tree = None
        zone_dom_comparison = None
        heading_dom_comparison = None
        if self.dom_llm_generator is not None:
            dom_tree = self.dom_llm_generator.generate(
                LLMTreeRequest(screenshot_path=screenshot_path, html=html)
            )
            llm_dom_tree = dom_tree
            zone_dom_comparison = compute_comparison(human_trees.zone_tree, dom_tree)
            heading_dom_comparison = compute_comparison(human_trees.heading_tree, dom_tree)
        llm_html_tree = None
        zone_html_comparison = None
        heading_html_comparison = None
        if self.html_llm_generator is not None:
            html_tree = self.html_llm_generator.generate(
                LLMTreeRequest(screenshot_path=screenshot_path, html=html)
            )
            llm_html_tree = html_tree
            zone_html_comparison = compute_comparison(human_trees.zone_tree, html_tree)
            heading_html_comparison = compute_comparison(human_trees.heading_tree, html_tree)
        llm_html_only_tree = None
        zone_html_only_comparison = None
        heading_html_only_comparison = None
        if self.html_only_llm_generator is not None:
            html_only_tree = self.html_only_llm_generator.generate(
                LLMTreeRequest(screenshot_path=screenshot_path, html=html)
            )
            llm_html_only_tree = html_only_tree
            zone_html_only_comparison = compute_comparison(human_trees.zone_tree, html_only_tree)
            heading_html_only_comparison = compute_comparison(human_trees.heading_tree, html_only_tree)
        llm_full_tree = None
        zone_full_comparison = None
        heading_full_comparison = None
        if self.full_llm_generator is not None:
            full_tree = self.full_llm_generator.generate(
                LLMTreeRequest(screenshot_path=screenshot_path, html=html)
            )
            llm_full_tree = full_tree
            zone_full_comparison = compute_comparison(human_trees.zone_tree, full_tree)
            heading_full_comparison = compute_comparison(human_trees.heading_tree, full_tree)
        return AnalysisResult(
            url=url,
            screenshot_path=screenshot_path,
            html_path=html_path,
            human_zone_tree=human_trees.zone_tree,
            human_heading_tree=human_trees.heading_tree,
            llm_tree=llm_tree,
            zone_comparison=zone_comparison,
            heading_comparison=heading_comparison,
            llm_dom_tree=llm_dom_tree,
            zone_dom_comparison=zone_dom_comparison,
            heading_dom_comparison=heading_dom_comparison,
            llm_html_tree=llm_html_tree,
            zone_html_comparison=zone_html_comparison,
            heading_html_comparison=heading_html_comparison,
            llm_html_only_tree=llm_html_only_tree,
            zone_html_only_comparison=zone_html_only_comparison,
            heading_html_only_comparison=heading_html_only_comparison,
            llm_full_tree=llm_full_tree,
            zone_full_comparison=zone_full_comparison,
            heading_full_comparison=heading_full_comparison,
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
        analyses = list(analyses)
        zone_metrics: List[dict] = []
        heading_metrics: List[dict] = []
        zone_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}
        heading_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}

        zone_dom_metrics: List[dict] = []
        heading_dom_metrics: List[dict] = []
        zone_dom_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}
        heading_dom_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}

        zone_html_metrics: List[dict] = []
        heading_html_metrics: List[dict] = []
        zone_html_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}
        heading_html_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}

        zone_html_only_metrics: List[dict] = []
        heading_html_only_metrics: List[dict] = []
        zone_html_only_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}
        heading_html_only_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}

        zone_full_metrics: List[dict] = []
        heading_full_metrics: List[dict] = []
        zone_full_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}
        heading_full_mismatch = {"missing": 0, "extra": 0, "depth_shift": 0, "order": 0}

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

            if analysis.zone_dom_comparison and analysis.heading_dom_comparison:
                zdom = analysis.zone_dom_comparison.metrics.flat()
                hdom = analysis.heading_dom_comparison.metrics.flat()
                zone_dom_metrics.append(zdom)
                heading_dom_metrics.append(hdom)

                zdm = analysis.zone_dom_comparison.metrics.mismatch_patterns
                hdm = analysis.heading_dom_comparison.metrics.mismatch_patterns
                zone_dom_mismatch["missing"] += zdm["missing_nodes"]["count"]
                zone_dom_mismatch["extra"] += zdm["extra_nodes"]["count"]
                zone_dom_mismatch["depth_shift"] += zdm["depth_shift"]["count"]
                zone_dom_mismatch["order"] += zdm["reading_order"]["gaps"]

                heading_dom_mismatch["missing"] += hdm["missing_nodes"]["count"]
                heading_dom_mismatch["extra"] += hdm["extra_nodes"]["count"]
                heading_dom_mismatch["depth_shift"] += hdm["depth_shift"]["count"]
                heading_dom_mismatch["order"] += hdm["reading_order"]["gaps"]

            if analysis.zone_html_comparison and analysis.heading_html_comparison:
                zhtml = analysis.zone_html_comparison.metrics.flat()
                hhtml = analysis.heading_html_comparison.metrics.flat()
                zone_html_metrics.append(zhtml)
                heading_html_metrics.append(hhtml)

                zhm = analysis.zone_html_comparison.metrics.mismatch_patterns
                hhm = analysis.heading_html_comparison.metrics.mismatch_patterns
                zone_html_mismatch["missing"] += zhm["missing_nodes"]["count"]
                zone_html_mismatch["extra"] += zhm["extra_nodes"]["count"]
                zone_html_mismatch["depth_shift"] += zhm["depth_shift"]["count"]
                zone_html_mismatch["order"] += zhm["reading_order"]["gaps"]

                heading_html_mismatch["missing"] += hhm["missing_nodes"]["count"]
                heading_html_mismatch["extra"] += hhm["extra_nodes"]["count"]
                heading_html_mismatch["depth_shift"] += hhm["depth_shift"]["count"]
                heading_html_mismatch["order"] += hhm["reading_order"]["gaps"]

            if analysis.zone_html_only_comparison and analysis.heading_html_only_comparison:
                zhtml_only = analysis.zone_html_only_comparison.metrics.flat()
                hhtml_only = analysis.heading_html_only_comparison.metrics.flat()
                zone_html_only_metrics.append(zhtml_only)
                heading_html_only_metrics.append(hhtml_only)

                zho = analysis.zone_html_only_comparison.metrics.mismatch_patterns
                hho = analysis.heading_html_only_comparison.metrics.mismatch_patterns
                zone_html_only_mismatch["missing"] += zho["missing_nodes"]["count"]
                zone_html_only_mismatch["extra"] += zho["extra_nodes"]["count"]
                zone_html_only_mismatch["depth_shift"] += zho["depth_shift"]["count"]
                zone_html_only_mismatch["order"] += zho["reading_order"]["gaps"]

                heading_html_only_mismatch["missing"] += hho["missing_nodes"]["count"]
                heading_html_only_mismatch["extra"] += hho["extra_nodes"]["count"]
                heading_html_only_mismatch["depth_shift"] += hho["depth_shift"]["count"]
                heading_html_only_mismatch["order"] += hho["reading_order"]["gaps"]

            if analysis.zone_full_comparison and analysis.heading_full_comparison:
                zfull = analysis.zone_full_comparison.metrics.flat()
                hfull = analysis.heading_full_comparison.metrics.flat()
                zone_full_metrics.append(zfull)
                heading_full_metrics.append(hfull)

                zfm = analysis.zone_full_comparison.metrics.mismatch_patterns
                hfm = analysis.heading_full_comparison.metrics.mismatch_patterns
                zone_full_mismatch["missing"] += zfm["missing_nodes"]["count"]
                zone_full_mismatch["extra"] += zfm["extra_nodes"]["count"]
                zone_full_mismatch["depth_shift"] += zfm["depth_shift"]["count"]
                zone_full_mismatch["order"] += zfm["reading_order"]["gaps"]

                heading_full_mismatch["missing"] += hfm["missing_nodes"]["count"]
                heading_full_mismatch["extra"] += hfm["extra_nodes"]["count"]
                heading_full_mismatch["depth_shift"] += hfm["depth_shift"]["count"]
                heading_full_mismatch["order"] += hfm["reading_order"]["gaps"]

        def _average(metrics_list: List[dict]) -> dict:
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            return {
                key: sum(m[key] for m in metrics_list) / len(metrics_list)
                for key in keys
            }

        summary = {
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

        if zone_dom_metrics and heading_dom_metrics:
            summary["zone_dom"] = {
                "average_metrics": _average(zone_dom_metrics),
                "mismatch_totals": zone_dom_mismatch,
                "count": len(zone_dom_metrics),
            }
            summary["heading_dom"] = {
                "average_metrics": _average(heading_dom_metrics),
                "mismatch_totals": heading_dom_mismatch,
                "count": len(heading_dom_metrics),
            }

        if zone_html_metrics and heading_html_metrics:
            summary["zone_html"] = {
                "average_metrics": _average(zone_html_metrics),
                "mismatch_totals": zone_html_mismatch,
                "count": len(zone_html_metrics),
            }
            summary["heading_html"] = {
                "average_metrics": _average(heading_html_metrics),
                "mismatch_totals": heading_html_mismatch,
                "count": len(heading_html_metrics),
            }

        if zone_html_only_metrics and heading_html_only_metrics:
            summary["zone_html_only"] = {
                "average_metrics": _average(zone_html_only_metrics),
                "mismatch_totals": zone_html_only_mismatch,
                "count": len(zone_html_only_metrics),
            }
            summary["heading_html_only"] = {
                "average_metrics": _average(heading_html_only_metrics),
                "mismatch_totals": heading_html_only_mismatch,
                "count": len(heading_html_only_metrics),
            }

        if zone_full_metrics and heading_full_metrics:
            summary["zone_full"] = {
                "average_metrics": _average(zone_full_metrics),
                "mismatch_totals": zone_full_mismatch,
                "count": len(zone_full_metrics),
            }
            summary["heading_full"] = {
                "average_metrics": _average(heading_full_metrics),
                "mismatch_totals": heading_full_mismatch,
                "count": len(heading_full_metrics),
            }

        return summary

    def visualize(
        self,
        analysis: AnalysisResult,
        *,
        zone_side_by_side_path: Optional[Path] = None,
        heading_side_by_side_path: Optional[Path] = None,
        zone_path: Optional[Path] = None,
        heading_path: Optional[Path] = None,
        llm_path: Optional[Path] = None,
        zone_dom_side_by_side_path: Optional[Path] = None,
        heading_dom_side_by_side_path: Optional[Path] = None,
        llm_dom_path: Optional[Path] = None,
        zone_html_side_by_side_path: Optional[Path] = None,
        heading_html_side_by_side_path: Optional[Path] = None,
        llm_html_path: Optional[Path] = None,
        zone_full_side_by_side_path: Optional[Path] = None,
        heading_full_side_by_side_path: Optional[Path] = None,
        llm_full_path: Optional[Path] = None,
        zone_html_only_side_by_side_path: Optional[Path] = None,
        heading_html_only_side_by_side_path: Optional[Path] = None,
        llm_html_only_path: Optional[Path] = None,
        with_clues: bool = False,
    ) -> None:
        if zone_side_by_side_path:
            plot_side_by_side(
                analysis.zone_comparison.human_tree,
                analysis.zone_comparison.llm_tree,
                path=zone_side_by_side_path,
                with_clues=with_clues,
            )
        if heading_side_by_side_path:
            plot_side_by_side(
                analysis.heading_comparison.human_tree,
                analysis.heading_comparison.llm_tree,
                path=heading_side_by_side_path,
                with_clues=with_clues,
            )
        if zone_path:
            plot_tree(analysis.human_zone_tree, title="Human Zone Tree", path=zone_path, with_clues=with_clues)
        if heading_path:
            plot_tree(analysis.human_heading_tree, title="Human Heading Tree", path=heading_path, with_clues=with_clues)
        if llm_path:
            plot_tree(analysis.llm_tree, title="LLM Tree (Vision)", path=llm_path, with_clues=with_clues)

        if (
            analysis.zone_dom_comparison
            and analysis.heading_dom_comparison
            and analysis.llm_dom_tree is not None
        ):
            if zone_dom_side_by_side_path:
                plot_side_by_side(
                    analysis.zone_dom_comparison.human_tree,
                    analysis.zone_dom_comparison.llm_tree,
                    path=zone_dom_side_by_side_path,
                    with_clues=with_clues,
                )
            if heading_dom_side_by_side_path:
                plot_side_by_side(
                    analysis.heading_dom_comparison.human_tree,
                    analysis.heading_dom_comparison.llm_tree,
                    path=heading_dom_side_by_side_path,
                    with_clues=with_clues,
                )
            if llm_dom_path:
                plot_tree(analysis.llm_dom_tree, title="LLM Tree (Vision + DOM)", path=llm_dom_path, with_clues=with_clues)

        if (
            analysis.zone_html_comparison
            and analysis.heading_html_comparison
            and analysis.llm_html_tree is not None
        ):
            if zone_html_side_by_side_path:
                plot_side_by_side(
                    analysis.zone_html_comparison.human_tree,
                    analysis.zone_html_comparison.llm_tree,
                    path=zone_html_side_by_side_path,
                    with_clues=with_clues,
                )
            if heading_html_side_by_side_path:
                plot_side_by_side(
                    analysis.heading_html_comparison.human_tree,
                    analysis.heading_html_comparison.llm_tree,
                    path=heading_html_side_by_side_path,
                    with_clues=with_clues,
                )
            if llm_html_path:
                plot_tree(analysis.llm_html_tree, title="LLM Tree (Vision + HTML)", path=llm_html_path, with_clues=with_clues)

        if (
            analysis.zone_html_only_comparison
            and analysis.heading_html_only_comparison
            and analysis.llm_html_only_tree is not None
        ):
            if zone_html_only_side_by_side_path:
                plot_side_by_side(
                    analysis.zone_html_only_comparison.human_tree,
                    analysis.zone_html_only_comparison.llm_tree,
                    path=zone_html_only_side_by_side_path,
                    with_clues=with_clues,
                )
            if heading_html_only_side_by_side_path:
                plot_side_by_side(
                    analysis.heading_html_only_comparison.human_tree,
                    analysis.heading_html_only_comparison.llm_tree,
                    path=heading_html_only_side_by_side_path,
                    with_clues=with_clues,
                )
            if llm_html_only_path:
                plot_tree(
                    analysis.llm_html_only_tree,
                    title="LLM Tree (HTML Only)",
                    path=llm_html_only_path,
                    with_clues=with_clues,
                )

        if (
            analysis.zone_full_comparison
            and analysis.heading_full_comparison
            and analysis.llm_full_tree is not None
        ):
            if zone_full_side_by_side_path:
                plot_side_by_side(
                    analysis.zone_full_comparison.human_tree,
                    analysis.zone_full_comparison.llm_tree,
                    path=zone_full_side_by_side_path,
                    with_clues=with_clues,
                )
            if heading_full_side_by_side_path:
                plot_side_by_side(
                    analysis.heading_full_comparison.human_tree,
                    analysis.heading_full_comparison.llm_tree,
                    path=heading_full_side_by_side_path,
                    with_clues=with_clues,
                )
            if llm_full_path:
                plot_tree(
                    analysis.llm_full_tree,
                    title="LLM Tree (Vision + DOM + HTML)",
                    path=llm_full_path,
                    with_clues=with_clues,
                )
