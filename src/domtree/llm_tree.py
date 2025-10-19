"""Interfaces and reference implementations for obtaining the LLM-derived tree."""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image

from .human_tree import HumanTreeOptions, HumanTreeExtractor
from .schema import NodeMetadata
from .tree import TreeNode

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LLMTreeRequest:
    screenshot_path: Path
    html: Optional[str] = None
    prompt: Optional[str] = None


class LLMTreeGenerator(ABC):
    """Abstract base class for tree generators backed by different LLM providers."""

    @abstractmethod
    def generate(self, request: LLMTreeRequest) -> TreeNode:
        """Return the tree perceived by the LLM for the given request."""


@dataclasses.dataclass
class HeuristicLLMOptions:
    """Configuration for the heuristic LLM tree generator used for offline experiments."""

    max_depth: int = 4
    max_children: int = 6
    align_with_visual_density: bool = True
    human_tree_options: HumanTreeOptions = dataclasses.field(default_factory=HumanTreeOptions)


class HeuristicLLMTreeGenerator(LLMTreeGenerator):
    """Approximate an LLM response by simplifying the human tree with visual cues."""

    def __init__(self, options: Optional[HeuristicLLMOptions] = None):
        self.options = options or HeuristicLLMOptions()

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        if not request.screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {request.screenshot_path}")
        html = request.html
        if html is None:
            raise ValueError("HeuristicLLMTreeGenerator requires HTML content for now")

        human_tree = HumanTreeExtractor(html, options=self.options.human_tree_options).extract()
        visual_profile = self._analyze_screenshot(request.screenshot_path)
        logger.debug("Computed visual profile: %s", visual_profile)
        simplified = self._simplify_tree(human_tree, visual_profile)
        simplified.metadata.notes.setdefault("llm", {})
        simplified.metadata.notes["llm"].update({"generator": "heuristic", "visual_profile": visual_profile})
        return simplified

    def _analyze_screenshot(self, path: Path) -> dict:
        with Image.open(path) as img:
            converted = img.convert("RGB")
            aspect_ratio = img.width / img.height if img.height else 1.0
            thumbnail = converted.copy()
            thumbnail.thumbnail((64, 64))
            pixels = list(thumbnail.getdata())
        avg_brightness = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3 * 255)
        dominant_colors = self._cluster_colors(pixels, k=4)
        return {
            "avg_brightness": round(avg_brightness, 3),
            "dominant_colors": dominant_colors,
            "aspect_ratio": round(aspect_ratio, 3),
        }

    def _cluster_colors(self, pixels, k: int):
        # Simple histogram-based clustering to avoid heavy dependencies
        buckets = {}
        for pixel in pixels:
            bucket = tuple(channel // 51 * 51 for channel in pixel)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        sorted_buckets = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:k]
        return [tuple(int(channel) for channel in color) for color, _ in sorted_buckets]

    def _simplify_tree(self, human_tree: TreeNode, visual_profile: dict) -> TreeNode:
        max_children = self.options.max_children
        max_depth = self.options.max_depth
        brightness = visual_profile.get("avg_brightness", 0.5)

        def clamp_children(children):
            if len(children) <= max_children:
                return children
            return children[:max_children]

        def recurse(node: TreeNode, depth: int = 0) -> TreeNode:
            metadata = node.metadata.copy()
            new_node = TreeNode(
                name=node.name,
                label=node.label,
                metadata=metadata,
            )
            if depth >= max_depth:
                return new_node
            limited_children = clamp_children(node.children)
            for child in limited_children:
                new_node.add_child(recurse(child, depth + 1))
            # Inject an artificial confidence score influenced by brightness and depth
            confidence = max(0.1, min(0.95, brightness - depth * 0.05 + len(limited_children) * 0.02))
            new_node.metadata.notes.setdefault("llm", {})
            new_node.metadata.notes["llm"]["confidence"] = round(confidence, 3)
            return new_node

        return recurse(human_tree)


class StubLLMTreeGenerator(LLMTreeGenerator):
    """Placeholder generator that raises an informative error."""

    def generate(self, request: LLMTreeRequest) -> TreeNode:  # pragma: no cover - simple stub
        raise NotImplementedError(
            "StubLLMTreeGenerator cannot produce a tree. Provide a concrete LLM implementation."
        )
