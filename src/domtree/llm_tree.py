"""Interfaces and reference implementations for obtaining the LLM-derived tree."""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import base64
import json
import requests

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


# --------------------------- Ollama LLaMA vision backend ---------------------------

DEFAULT_VISION_PROMPT = """
You are a meticulous document analyst. Using ONLY the provided webpage screenshot (ignore HTML) infer the human-perceived layout.
Return ONLY valid JSON matching this schema:
{
  "name": "zone|section|paragraph|list|table|figure|...",
  "label": "string",
  "metadata": {
    "type": "zone|section|paragraph|list|table|figure",
    "role": "main|sidebar|nav|toc|ad|body|...",
    "reading_order": <integer>,
    "text_heading": "optional heading text",
    "heading_level": <optional integer>,
    "text_preview": "optional excerpt",
    "dom_refs": ["css selector", ...],
    "vis_cues": {"bbox": [top,left,bottom,right]}
  },
  "children": [ ... recursive ... ]
}
Rules:
- Focus on what is visible in the screenshot. Ignore off-screen DOM sections.
- Create 1..N root-level zones (main content, sidebar, navigation, etc.).
- Under zones, add sections (heading hierarchy) and content blocks (paragraph, list, table, figure).
- Keep reading_order sequential within the visible flow (left to right, top to bottom).
- If unsure about text, leave text_preview empty instead of guessing.
Return nothing except the JSON.

""".strip()


@dataclasses.dataclass
class OllamaVisionOptions:
    endpoint: str = "http://localhost:11403/api/chat"
    model: str = "llama3.2-vision:11b"
    prompt_template: str = DEFAULT_VISION_PROMPT
    temperature: float = 0.1
    max_tokens: int = 2048


class OllamaVisionLLMTreeGenerator(LLMTreeGenerator):
    """Generate trees using Ollama's LLaMA 3.2 Vision 11B model."""

    def __init__(self, options: Optional[OllamaVisionOptions] = None):
        self.options = options or OllamaVisionOptions()

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        if not request.screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {request.screenshot_path}")

        prompt = self.options.prompt_template

        image_b64 = self._encode_image(request.screenshot_path)
        response = self._call_ollama(prompt, image_b64)
        return self._parse_response(response)

    def _encode_image(self, path: Path) -> str:
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def _call_ollama(self, prompt: str, image_b64: str) -> dict:
        payload = {
            "model": self.options.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You return only JSON with the specified schema.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_b64},
                    ],
                },
            ],
            "options": {
                "temperature": self.options.temperature,
                "num_predict": self.options.max_tokens,
            },
        }

        response = requests.post(self.options.endpoint, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        if "message" not in data or "content" not in data["message"]:
            raise ValueError(f"Unexpected Ollama response: {data}")
        return data["message"]

    def _parse_response(self, message: dict) -> TreeNode:
        parts = message.get("content", [])
        texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
        raw_text = "\n".join(texts).strip()
        if not raw_text:
            raise ValueError("Ollama response did not contain text")

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode Ollama JSON response: {raw_text[:200]}...") from exc

        return TreeNode.from_dict(data)
