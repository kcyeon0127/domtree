"""Interfaces and reference implementations for obtaining the LLM-derived tree."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

import base64
import json
import requests

from jsonschema import Draft7Validator, ValidationError

from PIL import Image

from .human_tree import HumanTreeOptions, HumanTreeExtractor
from .schema import NodeMetadata, TREE_JSON_SCHEMA
from .tree import TreeNode

logger = logging.getLogger(__name__)

SCHEMA_PROMPT_TEXT = json.dumps(TREE_JSON_SCHEMA, ensure_ascii=False, indent=2)
TREE_JSON_VALIDATOR = Draft7Validator(TREE_JSON_SCHEMA)


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

DEFAULT_VISION_PROMPT = f"""
You are a meticulous document analyst. Using ONLY the provided webpage screenshot (ignore HTML) infer the human-perceived layout.

You MUST return ONE valid JSON object that strictly follows the JSON Schema below. Do not include any additional commentary.

JSON Schema (do not modify):
{SCHEMA_PROMPT_TEXT}

Structural requirements:
- Create explicit zones for main content, sidebar, navigation, footer, etc. if visible.
- Within each zone, add sections that correspond to headings or visually distinct blocks.
- Under each section, include paragraph/list/table/figure nodes for major content blocks. Avoid collapsing multiple paragraphs into one node.
- Include reading_order for every node in visible order (top-left → bottom-right).
- When lists or tables are visible, represent them as separate nodes with suitable children if necessary.
- Aim to capture at least 20 nodes when the screenshot contains sufficient content.

Rules:
- Focus only on what is visible in the screenshot. Ignore off-screen DOM sections.
- Use concrete text snippets from the screenshot (truncate sensibly) or leave fields empty if unreadable.
- If unsure about text, leave text_preview empty instead of guessing.

Return nothing except the JSON.
Return a single JSON object without code fences.
Do NOT output schema examples, placeholders (e.g., "zone|section|..."), or explanatory text.
""".strip()


@dataclasses.dataclass
class OllamaVisionOptions:
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "llama3.2-vision:11b"
    prompt_template: str = DEFAULT_VISION_PROMPT
    temperature: float = 0.1
    max_tokens: int = 2048
    response_format: Optional[str] = "json"
    max_retries: int = 3
    template_markers: tuple[str, ...] = (
        "|section|",
        "zone|section|paragraph|list|table|figure",
        "main|sidebar",
        "optional heading",
        "schema",
        "template",
        "example",
        "placeholder",
    )
    min_total_nodes: int = 20


class OllamaVisionLLMTreeGenerator(LLMTreeGenerator):
    """Generate trees using Ollama's LLaMA 3.2 Vision 11B model."""

    def __init__(self, options: Optional[OllamaVisionOptions] = None):
        self.options = options or OllamaVisionOptions()

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        if not request.screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {request.screenshot_path}")

        image_b64 = self._encode_image(request.screenshot_path)
        base_prompt = request.prompt or self.options.prompt_template
        corrections: list[str] = [self._negative_guards()]
        last_response = ""
        last_reason = "json_decode"

        for attempt in range(1, self.options.max_retries + 1):
            full_prompt = self._compose_prompt(base_prompt, corrections)
            response = self._call_ollama(full_prompt, image_b64)
            last_response = response

            if self._looks_like_template_text(response):
                logger.debug("Attempt %s: template-like response detected", attempt)
                corrections.append(self._template_feedback())
                last_reason = "template_detected"
                continue

            try:
                data = self._extract_json_dict(response)
            except ValueError as exc:
                logger.debug("Attempt %s: JSON decode failed (%s)", attempt, exc)
                corrections.append(self._json_fix_feedback(str(exc)))
                last_reason = "json_decode"
                continue

            if self._looks_like_template_json(data):
                logger.debug("Attempt %s: JSON contains template markers", attempt)
                corrections.append(self._template_feedback())
                last_reason = "template_detected"
                continue

            try:
                self._validate_tree_dict(data)
            except ValueError as exc:
                logger.debug("Attempt %s: JSON schema validation failed (%s)", attempt, exc)
                corrections.append(self._validation_feedback(str(exc)))
                last_reason = "schema_validation"
                continue

            node_count = self._count_nodes(data)
            if node_count < self.options.min_total_nodes:
                logger.debug(
                    "Attempt %s: Tree too small (%s nodes < min %s)",
                    attempt,
                    node_count,
                    self.options.min_total_nodes,
                )
                corrections.append(self._detail_feedback(node_count))
                last_reason = "insufficient_detail"
                continue

            tree = TreeNode.from_dict(data)
            self._attach_raw_response(
                tree,
                response,
                attempts=attempt,
                prompt_hash=self._hash_prompt(full_prompt),
                status="ok",
            )
            tree.metadata.notes.setdefault("llm", {})
            tree.metadata.notes["llm"]["node_count"] = node_count
            return tree

        logger.warning("Ollama Vision failed after %s attempts: %s", self.options.max_retries, last_reason)
        error_tree = self._build_error_tree(
            last_response,
            reason=last_reason,
            attempts=self.options.max_retries,
        )
        final_prompt = self._compose_prompt(base_prompt, corrections)
        self._attach_raw_response(
            error_tree,
            last_response,
            attempts=self.options.max_retries,
            prompt_hash=self._hash_prompt(final_prompt),
            status=last_reason,
        )
        return error_tree

    def _encode_image(self, path: Path) -> str:
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def _call_ollama(self, prompt: str, image_b64: str) -> str:
        payload = {
            "model": self.options.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": self.options.temperature,
                "num_predict": self.options.max_tokens,
            },
        }

        if self.options.response_format:
            payload["format"] = self.options.response_format

        response = requests.post(self.options.endpoint, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        raw_text = data.get("response", "").strip()
        if not raw_text:
            raise ValueError(f"Unexpected Ollama response: {data}")
        return raw_text

    def _extract_json_dict(self, raw_text: str) -> dict:
        for candidate in self._candidate_json_strings(raw_text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        preview = raw_text[:200].replace("\n", " ")
        raise ValueError(f"Failed to decode JSON from response: {preview}...")

    def _candidate_json_strings(self, raw_text: str) -> Iterable[str]:
        text = raw_text.strip()
        candidates = [text]

        if text.startswith("```"):
            fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            fenced = re.sub(r"\s*```$", "", fenced).strip()
            if fenced:
                candidates.append(fenced)

        brace_candidate = self._extract_braced_json(text)
        if brace_candidate:
            candidates.append(brace_candidate)

        # Preserve order while removing duplicates
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                yield candidate

    @staticmethod
    def _extract_braced_json(text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1].strip()

    def _attach_raw_response(
        self,
        tree: TreeNode,
        raw_text: str,
        *,
        attempts: int,
        prompt_hash: str,
        status: str,
    ) -> None:
        preview = self._truncate(raw_text)
        notes = tree.metadata.notes.setdefault("llm", {})
        notes.setdefault("status", status)
        notes["raw_response_preview"] = preview
        notes["attempts"] = attempts
        notes["prompt_hash"] = prompt_hash
        tree.attributes.setdefault("llm", {})
        tree.attributes["llm"].update(
            {
                "raw_response": raw_text,
                "prompt_hash": prompt_hash,
                "attempts": attempts,
            }
        )

    def _build_error_tree(self, raw_text: str, *, reason: str, attempts: int) -> TreeNode:
        notes = {
            "llm": {
                "error": "invalid_json",
                "reason": reason,
                "attempts": attempts,
            }
        }
        metadata = NodeMetadata(
            node_type="llm_error",
            text_heading="LLM JSON 파싱 실패",
            text_preview=self._truncate(raw_text, limit=400),
            notes=notes,
        )
        attributes = {"llm": {"raw_response": raw_text}}
        return TreeNode(name="llm_error", label="LLM Parse Failure", metadata=metadata, attributes=attributes)

    def _looks_like_template_text(self, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in self.options.template_markers)

    def _looks_like_template_json(self, data) -> bool:
        if isinstance(data, dict):
            for value in data.values():
                if self._looks_like_template_json(value):
                    return True
        elif isinstance(data, list):
            return any(self._looks_like_template_json(item) for item in data)
        elif isinstance(data, str):
            return self._contains_marker(data)
        return False

    def _contains_marker(self, value: str) -> bool:
        lowered = value.lower()
        return any(marker in lowered for marker in self.options.template_markers)

    def _validate_tree_dict(self, data: dict) -> None:
        try:
            TREE_JSON_VALIDATOR.validate(data)
        except ValidationError as exc:
            raise ValueError(exc.message)

    def _negative_guards(self) -> str:
        return (
            "\n\nSYSTEM RULES:\n"
            "Return ONE valid JSON object only.\n"
            "Do NOT include code fences, explanations, schemas, or placeholders such as 'zone|section|...'.\n"
            "Use concrete values extracted from the screenshot.\n"
            "Represent every major visible block (zones → sections → content nodes)."
        )

    @staticmethod
    def _template_feedback() -> str:
        return (
            "\n\nSYSTEM CORRECTION: You returned a schema/template. Output actual JSON data only with real observations from the screenshot."
        )

    @staticmethod
    def _json_fix_feedback(error: str) -> str:
        return (
            f"\n\nSYSTEM CORRECTION: The previous output was not valid JSON ({error}). Return a single JSON object only, no comments or prose."
        )

    @staticmethod
    def _validation_feedback(error: str) -> str:
        return (
            f"\n\nSYSTEM CORRECTION: The JSON violated the schema ({error}). Return data that satisfies every required field in the schema."
        )

    def _compose_prompt(self, base_prompt: str, corrections: Iterable[str]) -> str:
        return base_prompt + "".join(corrections)

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _truncate(text: str, *, limit: int = 1200) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _count_nodes(data: dict) -> int:
        count = 1
        children = data.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    count += OllamaVisionLLMTreeGenerator._count_nodes(child)
        return count

    @staticmethod
    def _detail_feedback(node_count: int) -> str:
        return (
            f"\n\nSYSTEM CORRECTION: The previous JSON contained only {node_count} nodes."
            " Break down the layout into more zones, sections, and paragraph/list/table nodes to capture visible details."
        )
