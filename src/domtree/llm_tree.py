"""Interfaces and reference implementations for obtaining the LLM-derived tree."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from html import escape
from pathlib import Path
from typing import Iterable, Optional

import base64
import json
import requests

from bs4 import BeautifulSoup
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

        human_trees = HumanTreeExtractor(html, options=self.options.human_tree_options).extract()
        human_tree = human_trees.zone_tree
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
You are a world-class computer vision expert specializing in UI layout analysis. Your goal is to analyze the provided screenshot and generate a **highly detailed and granular** structural tree in JSON format.

**You MUST follow this two-step process:**
1.  **Step 1: Think First.** You must first fill out the `internal_thought_process` field. Follow the 'Step-by-Step Thinking Guide' below. This is mandatory.
2.  **Step 2: Generate Tree.** After, and only after, completing your thought process, use your analysis to generate the tree structure under the `children` field.

---

### Step-by-Step Thinking Guide (for `internal_thought_process`)

1.  **Scan for Atomic Elements:** Start with a bottom-up scan. List all the smallest, individual elements you can see (e.g., buttons, text inputs, individual links, icons, labels, small text blocks).
2.  **Apply Critical Rules:**
    *   **DO NOT MERGE:** Never merge visually distinct elements into one. If you see 5 links in a menu, list 5 link elements. If you see 3 social media icons, list 3 icon elements. Be specific.
    *   **BE GRANULAR:** Prefer more detailed nodes over fewer, abstract ones.
3.  **Group Elements:** Group the identified atomic elements into logical components (e.g., a 'login form' component containing labels, inputs, and a button).
4.  **Define Zones:** Group the components into major page zones (e.g., `zone_header`, `zone_main`, `zone_footer`).
5.  **Plan Hierarchy:** Briefly outline the final parent-child hierarchy you will build based on this analysis.

---

### Example

**Example `internal_thought_process`:**
"1. Atomic elements: I see a 'Username' label, a text input box, a 'Password' label, another text input box, and a 'Login' button.\n2. Rules: I will not merge these. Each will be a separate node.\n3. Grouping: These 5 elements form a 'Login Form'.\n4. Zoning: This form is inside the 'Main Content' zone.\n5. Plan: I will create a `section_login_form` node with 5 children: label, input, label, input, button."

**The resulting `children` array would then be structured based on that plan.**

---

### JSON Schema to Follow

You MUST return ONE valid JSON object that strictly follows the JSON Schema below. Do not include any additional commentary outside the specified fields.

```json
{SCHEMA_PROMPT_TEXT}
```

### Field Requirements & Rules

- Use the exact field names from the schema (snake_case such as `text_heading`, `reading_order`, `dom_refs`, `vis_cues`, `text_preview`).
- If a field has no value, omit it entirely. For arrays use `[]`, for objects use `{{}}`. Never emit `null` for arrays/objects.
- `dom_refs` must be an array (even if empty). `vis_cues` must be an object with numeric `bbox` when available.
- Choose descriptive `name`/`type` values derived from the content (e.g., `zone_main`, `section_introduction`, `paragraph_overview`).
- Every node MUST include a `metadata` object containing at least `type`, `reading_order`, `dom_refs` (array), and `vis_cues` (object). Put text snippets in `metadata.text_preview`.
- For a visually complex page, aim to capture at least 20-25 nodes. For simpler pages, fewer nodes are acceptable.
- Tree depth should be appropriate to the content's complexity. Do not sacrifice necessary detail to meet an arbitrary depth limit.
- Return nothing except the single JSON object. Do NOT use code fences.
""".strip()


DEFAULT_HTML_ONLY_PROMPT = f"""
You are a meticulous document analyst. Using ONLY the cleaned viewport HTML snippet supplied below, infer the human-perceived layout.

You MUST return ONE valid JSON object that strictly follows the JSON Schema below. Do not include any additional commentary.

JSON Schema (do not modify):
{SCHEMA_PROMPT_TEXT}

Structural requirements:
- Identify zones for main content, navigation, sidebar, footer, etc. based on semantic hints in the HTML.
- Within each zone, create sections that correspond to headings or structural containers.
- Under each section, include paragraph/list/table/figure nodes for major content blocks.
- Include reading_order for every node following DOM order (top to bottom, left to right as implied by the snippet).
- Limit the tree depth to at most 4 levels by grouping related content as siblings instead of creating single-child chains.
- Aim to capture at least 10 nodes when the snippet contains sufficient content.

Field requirements:
- Use the exact field names from the schema (snake_case such as `text_heading`, `heading_level`, `reading_order`, `dom_refs`, `vis_cues`, `text_preview`).
- If a field has no value, omit it entirely. For arrays use `[]`, for objects use `{{}}`. Never emit `null` for arrays/objects.
- `dom_refs` must be an array (even if empty). `vis_cues` must be an object with numeric `bbox` when available.
- Choose descriptive `name`/`type` values derived from the HTML semantics (e.g., `zone_main`, `section_introduction`, `paragraph_overview`).
- Every node MUST include a `metadata` object containing at least `type`, `reading_order`, `dom_refs` (array), and `vis_cues` (object). Put text snippets in `metadata.text_preview`.
- Set `metadata.notes.llm.source` to `"html_only"` for every node.
- Provide at least 3 nodes. Use a `zone → section → paragraph` pattern as a minimum baseline, and increment `reading_order` globally (1,2,3...).
- `dom_refs` should reference visible DOM elements or remain an empty array `[]` when unknown.

Rules:
- Use only the HTML snippet for evidence. Do not invent nodes that are not represented in the markup.
- Prefer headings (`<h1>`-`<h6>`), ARIA roles, and structural tags to infer zones and sections.
- If text is long, place a truncated preview (≤80 characters) in `metadata.text_preview`.
- Return nothing except the JSON. Do NOT output schema examples, placeholders, or explanatory text.
""".strip()


@dataclasses.dataclass
class OllamaVisionOptions:
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "llama3.2-vision:11b"
    prompt_template: str = DEFAULT_VISION_PROMPT
    temperature: float = 0.1
    max_tokens: int = 4096
    response_format: Optional[str] = "json"
    max_retries: int = 5
    template_markers: tuple[str, ...] | None = None
    min_total_nodes: int = 3


@dataclasses.dataclass
class OllamaVisionDomOptions(OllamaVisionOptions):
    max_dom_chars: int = 2000
    max_sections: int = 40
    paragraphs_per_section: int = 2


class OllamaVisionLLMTreeGenerator(LLMTreeGenerator):
    """Generate trees using Ollama's LLaMA 3.2 Vision 11B model."""

    def __init__(self, options: Optional[OllamaVisionOptions] = None):
        self.options = options or OllamaVisionOptions()
        self._last_debug: dict = {}
        self._extra_debug: dict | None = None

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        if not request.screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {request.screenshot_path}")

        image_b64 = self._encode_image(request.screenshot_path)
        base_prompt = request.prompt or self.options.prompt_template
        corrections: list[str] = [self._negative_guards()]
        last_response = ""
        last_reason = "json_decode"
        self._last_debug = {
            "screenshot_path": str(request.screenshot_path),
            "image_b64_chars": len(image_b64),
            "image_b64_preview": image_b64[:80],
            "generator": self.__class__.__name__,
            "llm_model": getattr(self.options, "model", "unknown"),
        }
        if self._extra_debug:
            self._last_debug.update(self._extra_debug)
            self._extra_debug = None
        try:
            self._last_debug["image_bytes"] = request.screenshot_path.stat().st_size
        except OSError:
            pass

        for attempt in range(1, self.options.max_retries + 1):
            full_prompt = self._compose_prompt(base_prompt, corrections)
            self._last_debug.update(
                {
                    "attempt": attempt,
                    "prompt_chars": len(full_prompt),
                    "prompt_preview": self._truncate(full_prompt, limit=400),
                    "correction_count": max(0, len(corrections) - 1),
                }
            )
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
                corrections.append(self._detail_feedback(node_count, self.options.min_total_nodes))
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

        logger.warning("%s failed after %s attempts: %s", self.__class__.__name__, self.options.max_retries, last_reason)
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

        self._last_debug.update(
            {
                "backend": "ollama",
                "endpoint": self.options.endpoint,
                "model": self.options.model,
                "payload_keys": list(payload.keys()),
            }
        )

        response = requests.post(self.options.endpoint, json=payload, timeout=180)
        self._last_debug["status_code"] = response.status_code
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

        fenced = self._extract_fenced_json(text)
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
        candidate = text[start : end + 1]
        try:
            depth = 0
            for idx, char in enumerate(candidate):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0 and idx != len(candidate) - 1:
                        candidate = candidate[: idx + 1]
                        break
        except Exception:
            pass
        return candidate.strip()

    @staticmethod
    def _extract_fenced_json(text: str) -> Optional[str]:
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

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
        if self._last_debug:
            notes["debug"] = dict(self._last_debug)
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
        markers = self.options.template_markers
        if not markers:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in markers)

    def _looks_like_template_json(self, data) -> bool:
        markers = self.options.template_markers
        if not markers:
            return False
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
        markers = self.options.template_markers
        if not markers:
            return False
        lowered = value.lower()
        return any(marker in lowered for marker in markers)

    def _validate_tree_dict(self, data: dict) -> None:
        try:
            TREE_JSON_VALIDATOR.validate(data)
        except ValidationError as exc:
            raise ValueError(exc.message)

    def _negative_guards(self) -> str:
        min_nodes = self.options.min_total_nodes
        return (
            "\n\nSYSTEM RULES:\n"
            "Return ONE valid JSON object only.\n"
            "Do NOT include code fences, explanations, schemas, or placeholders such as 'zone|section|...'.\n"
            "Use concrete values extracted from the screenshot.\n"
            "Represent every major visible block (zones → sections → content nodes).\n"
            "Use schema field names exactly (snake_case) and avoid null for arrays/objects.\n"
            "Keep arrays short (<=5 items) and include only a single bbox array per node.\n"
            f"Ensure the tree contains at least {min_nodes} nodes covering the visible content."
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
            "\n\n**CRITICAL SYSTEM CORRECTION: YOUR PREVIOUS JSON OUTPUT FAILED SCHEMA VALIDATION.**\n\n"
            f"**Error:** {error}\n\n"
            "This is a critical failure. You MUST fix the structure. Pay close attention to the following common mistakes:\n"
            "- **MANDATORY FIELDS:** Every single node object, at every level, MUST have the keys: `\"name\"`, `\"metadata\"`, and `\"children\"`. The `metadata` object itself MUST have a `\"type\"` key.\n"
            "- **CORRECT DATA TYPES:** `children` MUST be an array (e.g., `[]`), even if there are no children. `metadata` MUST be an object (`{}`). `heading_level` and `reading_order` must be integers, not strings.\n"
            "- **NO EXTRA KEYS:** Do not add keys at the top level that are not in the schema (like `\"text\"` or `\"content\"`). All descriptive text goes inside `metadata.text_preview` or `label`.\n\n"
            "Review the original schema, identify your mistake based on the error message, and provide a corrected, valid JSON object. **DO NOT repeat the mistake. Return only the JSON object.**"
        )

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
    def _detail_feedback(node_count: int, target: int) -> str:
        return (
            f"\n\nSYSTEM CORRECTION: The previous JSON contained only {node_count} nodes, but at least {target} nodes are expected."
            " Break down the layout into more zones, sections, and paragraph/list/table nodes to capture visible details."
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


class OllamaVisionDomLLMTreeGenerator(OllamaVisionLLMTreeGenerator):
    """Generate trees using screenshot plus viewport DOM summary as context."""

    def __init__(self, options: Optional[OllamaVisionDomOptions] = None):
        super().__init__(options)
        self.options: OllamaVisionDomOptions = options or OllamaVisionDomOptions()

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        html = request.html
        if not html:
            return super().generate(request)

        summary = self._summarize_dom(html)
        preview = summary[:200]
        self._extra_debug = {
            "dom_summary_chars": len(summary),
            "dom_summary_preview": preview,
            "dom_summary_truncated": len(summary) > len(preview),
        }
        dom_prompt = (
            self.options.prompt_template
            + "\n\nDOM SUMMARY (viewport approximation):\n"
            + summary
            + "\n\nSOURCE GUIDANCE: Nodes supported by this DOM SUMMARY must label `metadata.notes.llm.source` as `dom_summary`."
            + " If the screenshot also confirms them, combine with `vision` (e.g., `vision+dom_summary`)."
        )
        patched_request = dataclasses.replace(request, prompt=dom_prompt)
        return super().generate(patched_request)

    def _summarize_dom(self, html: str) -> str:
        try:
            options = HumanTreeOptions(
                min_text_length=25,
                restrict_to_viewport=True,
                include_lists=True,
                include_tables=True,
                include_figures=False,
                include_text_nodes=False,
                max_list_items=5,
            )
            extractor = HumanTreeExtractor(html, options=options)
            bundle = extractor.extract()
            summary = self._summarize_zone_tree(bundle.zone_tree)
        except Exception as exc:  # fallback to simple heading summary
            logger.debug("Viewport DOM summarisation failed: %s", exc)
            summary = self._fallback_dom_summary(html)

        max_chars = max(256, self.options.max_dom_chars)
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."
        return summary or "(No visible viewport DOM extracted)"

    def _summarize_zone_tree(self, tree: TreeNode) -> str:
        lines: list[str] = []
        section_budget = self.options.max_sections

        for zone in tree.children:
            if section_budget <= 0:
                break
            role = zone.metadata.role or "zone"
            heading = zone.metadata.text_heading or zone.label
            bbox = zone.metadata.visual_cues.bbox
            bbox_text = (
                f"bbox={bbox}" if bbox and all(v is not None for v in bbox) else "bbox=?"
            )
            lines.append(f"ZONE[{zone.metadata.reading_order}] {role}: {heading} ({bbox_text})")

            for child in zone.children:
                if section_budget <= 0:
                    break
                if child.name != "section":
                    continue
                section_budget -= 1
                sec_heading = child.metadata.text_heading or child.label
                level = child.metadata.heading_level
                level_txt = f"L{level}" if level else ""
                lines.append(f"- SECTION{level_txt} [{child.metadata.reading_order}]: {sec_heading}")

                taken = 0
                for grand in child.children:
                    if taken >= self.options.paragraphs_per_section:
                        break
                    if grand.name not in {"paragraph", "list", "table", "figure"}:
                        continue
                    preview = (grand.metadata.text_preview or grand.label or "").strip()
                    if not preview:
                        continue
                    label = grand.name.upper()
                    lines.append(f"  - {label}: {preview}")
                    taken += 1

            # If the zone had no sections, fall back to direct content summaries
            if not any(child.name == "section" for child in zone.children):
                taken = 0
                for child in zone.children:
                    if taken >= self.options.paragraphs_per_section:
                        break
                    preview = (child.metadata.text_preview or child.label or "").strip()
                    if not preview:
                        continue
                    lines.append(f"- CONTENT [{child.metadata.reading_order}]: {preview}")
                    taken += 1

        return "\n".join(lines)

    def _fallback_dom_summary(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        body = soup.body or soup
        lines: list[str] = []

        headings = body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        for heading in headings[: self.options.max_sections]:
            text = heading.get_text(" ", strip=True)
            if not text:
                continue
            level = heading.name.upper()
            lines.append(f"{level}: {text}")

            if self.options.paragraphs_per_section <= 0:
                continue
            collected = 0
            sibling = heading.find_next_sibling()
            while sibling and collected < self.options.paragraphs_per_section:
                sibling_name = getattr(sibling, "name", "").lower() if getattr(sibling, "name", None) else ""
                if sibling_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    break
                snippet = sibling.get_text(" ", strip=True) if getattr(sibling, "get_text", None) else str(sibling).strip()
                if snippet:
                    lines.append(f"- {snippet}")
                    collected += 1
                sibling = sibling.find_next_sibling()

        return "\n".join(lines)


# --------------------------- OpenRouter GPT-4o mini backend ---------------------------


@dataclasses.dataclass
class OpenRouterVisionOptions(OllamaVisionOptions):
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "openai/gpt-4o-mini"
    api_key: str = dataclasses.field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "sample"))
    referer: str = "https://example.com/domtree"
    title: str = "DOMTree Analyzer"
    response_format: Optional[str] = "json_object"


@dataclasses.dataclass
class OpenRouterVisionDomOptions(OpenRouterVisionOptions):
    max_dom_chars: int = 2000
    max_sections: int = 40
    paragraphs_per_section: int = 2


@dataclasses.dataclass
class OpenRouterVisionHtmlOptions(OpenRouterVisionOptions):
    max_html_chars: int = 4000
    max_sections: int = 40
    paragraphs_per_section: int = 3
    text_preview_limit: int = 160


@dataclasses.dataclass
class OpenRouterVisionFullOptions(OpenRouterVisionOptions):
    max_dom_chars: int = 2000
    max_html_chars: int = 4000
    max_sections: int = 40
    paragraphs_per_section: int = 3
    text_preview_limit: int = 160


@dataclasses.dataclass
class OpenRouterHtmlOnlyOptions(OpenRouterVisionHtmlOptions):
    prompt_template: str = DEFAULT_HTML_ONLY_PROMPT


class OpenRouterVisionLLMTreeGenerator(OllamaVisionLLMTreeGenerator):
    """Generate trees using OpenRouter's ChatGPT 4o mini vision capabilities."""

    def __init__(self, options: Optional[OpenRouterVisionOptions] = None):
        options = options or OpenRouterVisionOptions()
        super().__init__(options)
        self.options = options

    def _call_ollama(self, prompt: str, image_b64: str) -> str:  # type: ignore[override]
        if not self.options.api_key:
            raise ValueError("OpenRouter API key is not configured. Set OPENROUTER_API_KEY or update the options.")
        if self.options.api_key == "sample":
            logger.warning("OpenRouter API key is set to placeholder 'sample'. Replace it with a real key before production use.")
        headers = {
            "Authorization": f"Bearer {self.options.api_key}",
            "Content-Type": "application/json",
        }
        if self.options.referer:
            headers["HTTP-Referer"] = self.options.referer
        if self.options.title:
            headers["X-Title"] = self.options.title

        self._last_debug.update(
            {
                "backend": "openrouter",
                "endpoint": self.options.endpoint,
                "model": self.options.model,
                "referer": bool(self.options.referer),
                "title": self.options.title,
            }
        )

        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]

        payload = {
            "model": self.options.model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a meticulous document analyst. Follow the user instructions exactly and answer in JSON only.",
                        }
                    ],
                },
                {"role": "user", "content": message_content},
            ],
            "temperature": self.options.temperature,
            "max_tokens": self.options.max_tokens,
        }

        if self.options.response_format:
            payload["response_format"] = {"type": self.options.response_format}

        self._last_debug.update(
            {
                "message_count": len(payload["messages"]),
                "has_image": any(part.get("type") == "input_image" for part in message_content),
                "response_format": self.options.response_format,
            }
        )

        response = requests.post(
            self.options.endpoint,
            headers=headers,
            json=payload,
            timeout=180,
        )
        self._last_debug["status_code"] = response.status_code
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"Unexpected OpenRouter response: {data}")

        message = choices[0].get("message", {})
        raw_text = self._extract_message_text(message)
        if not raw_text:
            raise ValueError(f"Empty response content from OpenRouter: {data}")
        self._last_debug["usage"] = data.get("usage", {})
        return raw_text.strip()

    @staticmethod
    def _extract_message_text(message: dict) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value")
                    if text:
                        fragments.append(str(text))
                elif isinstance(item, str):
                    fragments.append(item)
            return "".join(fragments).strip()
        return ""


class OpenRouterVisionDomLLMTreeGenerator(OpenRouterVisionLLMTreeGenerator):
    """OpenRouter vision generator with additional viewport DOM context."""

    def __init__(self, options: Optional[OpenRouterVisionDomOptions] = None):
        options = options or OpenRouterVisionDomOptions()
        super().__init__(options)
        self.options = options

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        html = request.html
        if not html:
            return super().generate(request)

        summary = self._summarize_dom(html)
        preview = summary[:200]
        self._extra_debug = {
            "dom_summary_chars": len(summary),
            "dom_summary_preview": preview,
            "dom_summary_truncated": len(summary) > len(preview),
        }
        dom_prompt = (
            self.options.prompt_template
            + "\n\nDOM SUMMARY (viewport approximation):\n"
            + summary
            + "\n\nSOURCE GUIDANCE: Nodes inferred from this DOM SUMMARY should set metadata.notes.llm.source to \"dom_summary\" (combine with \"vision\" when the screenshot confirms the same node)."
        )
        patched_request = dataclasses.replace(request, prompt=dom_prompt)
        return super().generate(patched_request)

    def _summarize_dom(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        body = soup.body or soup
        lines: list[str] = []

        headings = body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        for heading in headings[: self.options.max_sections]:
            text = heading.get_text(" ", strip=True)
            if not text:
                continue
            level = heading.name.upper()
            lines.append(f"{level}: {text}")

            if self.options.paragraphs_per_section <= 0:
                continue
            collected = 0
            sibling = heading.find_next_sibling()
            while sibling and collected < self.options.paragraphs_per_section:
                sibling_name = getattr(sibling, "name", "").lower() if getattr(sibling, "name", None) else ""
                if sibling_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    break
                snippet = sibling.get_text(" ", strip=True) if getattr(sibling, "get_text", None) else str(sibling).strip()
                if snippet:
                    lines.append(f"- {snippet}")
                    collected += 1
                sibling = sibling.find_next_sibling()

        summary = "\n".join(lines)
        max_chars = max(256, self.options.max_dom_chars)
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."
        return summary or "(No visible DOM text extracted)"


class OpenRouterVisionHtmlLLMTreeGenerator(OpenRouterVisionLLMTreeGenerator):
    """OpenRouter vision generator with cleaned viewport HTML outline as context."""

    def __init__(self, options: Optional[OpenRouterVisionHtmlOptions] = None):
        options = options or OpenRouterVisionHtmlOptions()
        super().__init__(options)
        self.options = options

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        html = request.html
        if not html:
            return super().generate(request)

        clean_html = self._build_viewport_html(html)
        preview = clean_html[:200]
        self._extra_debug = {
            "clean_html_chars": len(clean_html),
            "clean_html_preview": preview,
            "clean_html_truncated": len(clean_html) > len(preview),
        }
        html_prompt = (
            self.options.prompt_template
            + "\n\nVIEWPORT HTML (cleaned):\n"
            + clean_html
            + "\n\nSOURCE GUIDANCE: When a node is derived from this HTML outline, set metadata.notes.llm.source to \"html_outline\". Combine with \"vision\" if the screenshot confirms it."
        )
        patched_request = dataclasses.replace(request, prompt=html_prompt)
        return super().generate(patched_request)

    def _build_viewport_html(self, html: str) -> str:
        try:
            options = HumanTreeOptions(
                min_text_length=25,
                restrict_to_viewport=True,
                include_lists=True,
                include_tables=True,
                include_figures=True,
                include_text_nodes=False,
                max_list_items=5,
            )
            extractor = HumanTreeExtractor(html, options=options)
            bundle = extractor.extract()
            rendered = self._render_zone_tree_as_html(bundle.zone_tree)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Viewport HTML rendering failed: %s", exc)
            rendered = self._fallback_html(html)

        max_chars = max(256, self.options.max_html_chars)
        if len(rendered) > max_chars:
            rendered = rendered[: max_chars - 3] + "..."
        return rendered or "<document />"

    def _render_zone_tree_as_html(self, tree: TreeNode) -> str:
        lines = ["<document>"]
        section_budget = self.options.max_sections
        for zone in tree.children:
            if section_budget <= 0:
                break
            zone_lines, section_budget = self._render_zone(zone, section_budget, indent=1)
            if zone_lines:
                lines.extend(zone_lines)
        lines.append("</document>")
        return "\n".join(lines)

    def _render_zone(
        self,
        zone: TreeNode,
        section_budget: int,
        *,
        indent: int,
    ) -> tuple[list[str], int]:
        lines: list[str] = []
        indent_str = "  " * indent
        attrs = {
            "role": zone.metadata.role or zone.name,
            "heading": zone.metadata.text_heading or zone.label or "",
            "order": str(zone.metadata.reading_order),
        }
        bbox = zone.metadata.visual_cues.bbox
        if bbox and all(value is not None for value in bbox):
            attrs["bbox"] = ",".join(str(round(value, 2)) for value in bbox)
        attr_str = " ".join(
            f'{name}="{escape(value)}"'
            for name, value in attrs.items()
            if value
        )
        lines.append(f"{indent_str}<zone {attr_str}>")

        sections = [child for child in zone.children if child.name == "section"]
        if sections:
            for section in sections:
                if section_budget <= 0:
                    break
                section_lines, section_budget = self._render_section(
                    section,
                    section_budget,
                    indent=indent + 1,
                )
                if section_lines:
                    lines.extend(section_lines)
        else:
            content_lines = self._render_content_nodes(
                zone.children,
                limit=self.options.paragraphs_per_section,
                indent=indent + 1,
            )
            lines.extend(content_lines)

        lines.append(f"{indent_str}</zone>")
        return lines, section_budget

    def _render_section(
        self,
        section: TreeNode,
        section_budget: int,
        *,
        indent: int,
    ) -> tuple[list[str], int]:
        lines: list[str] = []
        indent_str = "  " * indent
        attrs = {
            "heading": section.metadata.text_heading or section.label or "",
            "level": str(section.metadata.heading_level or ""),
            "order": str(section.metadata.reading_order),
        }
        attr_str = " ".join(
            f'{name}="{escape(value)}"'
            for name, value in attrs.items()
            if value
        )
        lines.append(f"{indent_str}<section {attr_str}>")

        content_lines = self._render_content_nodes(
            section.children,
            limit=self.options.paragraphs_per_section,
            indent=indent + 1,
        )
        lines.extend(content_lines)

        lines.append(f"{indent_str}</section>")
        return lines, section_budget - 1

    def _render_content_nodes(
        self,
        nodes: Iterable[TreeNode],
        *,
        limit: int,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        taken = 0
        indent_str = "  " * indent
        for node in nodes:
            if limit >= 0 and taken >= limit:
                break
            if node.name == "section":
                continue
            text = self._short_text(node)
            if node.name == "list" and node.children:
                lines.append(f"{indent_str}<list order=\"{node.metadata.reading_order}\">")
                for item in node.children[: self.options.paragraphs_per_section]:
                    item_text = self._short_text(item)
                    if not item_text:
                        continue
                    lines.append(
                        f"{indent_str}  <item order=\"{item.metadata.reading_order}\">{escape(item_text)}</item>"
                    )
                lines.append(f"{indent_str}</list>")
                taken += 1
                continue
            if not text:
                continue
            tag = {
                "paragraph": "paragraph",
                "table": "table",
                "figure": "figure",
            }.get(node.name, "content")
            lines.append(
                f"{indent_str}<{tag} order=\"{node.metadata.reading_order}\">{escape(text)}</{tag}>"
            )
            taken += 1
        return lines

    def _short_text(self, node: TreeNode) -> str:
        text = node.metadata.text_preview or node.label or ""
        text = text.strip()
        if not text:
            return ""
        limit = max(32, self.options.text_preview_limit)
        if len(text) > limit:
            return text[: limit - 1] + "…"
        return text

    def _fallback_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        body = soup.body or soup
        text = body.get_text(" ", strip=True)
        text = (text or "No visible content").strip()
        limit = max(256, self.options.max_html_chars)
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return "<document>\n  <text>" + escape(text) + "</text>\n</document>"


class OpenRouterVisionFullLLMTreeGenerator(OpenRouterVisionLLMTreeGenerator):
    """OpenRouter vision generator using DOM outline and cleaned HTML together."""

    def __init__(self, options: Optional[OpenRouterVisionFullOptions] = None):
        options = options or OpenRouterVisionFullOptions()
        super().__init__(options)
        self.options = options
        self._dom_helper = OpenRouterVisionDomLLMTreeGenerator(
            options=OpenRouterVisionDomOptions(
                endpoint=options.endpoint,
                model=options.model,
                api_key=options.api_key,
                referer=options.referer,
                title=options.title,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                response_format=options.response_format,
                max_dom_chars=options.max_dom_chars,
                max_sections=options.max_sections,
                paragraphs_per_section=options.paragraphs_per_section,
            )
        )
        self._html_helper = OpenRouterVisionHtmlLLMTreeGenerator(
            options=OpenRouterVisionHtmlOptions(
                endpoint=options.endpoint,
                model=options.model,
                api_key=options.api_key,
                referer=options.referer,
                title=options.title,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                response_format=options.response_format,
                max_html_chars=options.max_html_chars,
                max_sections=options.max_sections,
                paragraphs_per_section=options.paragraphs_per_section,
                text_preview_limit=options.text_preview_limit,
            )
        )

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        html = request.html
        if not html:
            return super().generate(request)

        dom_summary = self._dom_helper._summarize_dom(html)
        html_outline = self._html_helper._build_viewport_html(html)

        preview_dom = dom_summary[:200]
        preview_html = html_outline[:200]
        self._extra_debug = {
            "dom_summary_chars": len(dom_summary),
            "dom_summary_preview": preview_dom,
            "dom_summary_truncated": len(dom_summary) > len(preview_dom),
            "clean_html_chars": len(html_outline),
            "clean_html_preview": preview_html,
            "clean_html_truncated": len(html_outline) > len(preview_html),
        }

        guideline = (
            "The following context blocks are provided in order. "
            "Use them all when building the JSON tree:"
            "\n1. DOM SUMMARY — high-level zone/section outline derived from viewport content."
            "\n2. VIEWPORT HTML — cleaned HTML snippet with detailed child nodes and text previews."
            "\nCross-check these with the screenshot and prefer visual evidence when there is any mismatch."
            "\nSet `metadata.notes.llm.source` to reflect which evidence was used: `vision`, `dom_summary`, `html_outline` (combine with `+` when multiple sources support the node)."
        )

        full_prompt = (
            self.options.prompt_template
            + "\n\nCONTEXT GUIDELINES:\n"
            + guideline
            + "\n\nDOM SUMMARY (viewport outline):\n"
            + dom_summary
            + "\n\nVIEWPORT HTML (cleaned snippet):\n"
            + html_outline
        )

        patched_request = dataclasses.replace(request, prompt=full_prompt)
        return super().generate(patched_request)


class OpenRouterHtmlOnlyLLMTreeGenerator(OpenRouterVisionHtmlLLMTreeGenerator):
    """Generate trees using only the cleaned viewport HTML snippet."""

    def __init__(self, options: Optional[OpenRouterHtmlOnlyOptions] = None):
        options = options or OpenRouterHtmlOnlyOptions()
        super().__init__(options)
        self.options = options

    def generate(self, request: LLMTreeRequest) -> TreeNode:
        html = request.html
        if not html:
            raise ValueError("HTML content is required for the HTML-only generator")

        clean_html = self._build_viewport_html(html)
        preview = clean_html[:200]
        self._extra_debug = {
            "clean_html_chars": len(clean_html),
            "clean_html_preview": preview,
            "clean_html_truncated": len(clean_html) > len(preview),
        }

        base_prompt = request.prompt or self.options.prompt_template
        corrections: list[str] = [self._negative_guards()]
        last_response = ""
        last_reason = "json_decode"
        self._last_debug = {
            "generator": self.__class__.__name__,
            "llm_model": getattr(self.options, "model", "unknown"),
            "backend": "openrouter_html_only",
            "clean_html_chars": len(clean_html),
            "clean_html_preview": preview,
        }

        for attempt in range(1, self.options.max_retries + 1):
            full_prompt = (
                base_prompt
                + "\n\nVIEWPORT HTML (cleaned snippet):\n"
                + clean_html
                + "\n\nSYSTEM NOTE: The snippet already reflects viewport filtering. Use headings, roles, and structure from this HTML to build the tree."
            )
            full_prompt = self._compose_prompt(full_prompt, corrections)
            response = self._call_openrouter_text_only(full_prompt)
            last_response = response

            try:
                data = self._extract_json_dict(response)
            except ValueError as exc:
                corrections.append(self._json_fix_feedback(str(exc)))
                last_reason = "json_decode"
                continue

            if self._looks_like_template_json(data):
                corrections.append(self._template_feedback())
                last_reason = "template_detected"
                continue

            try:
                self._validate_tree_dict(data)
            except ValueError as exc:
                corrections.append(self._validation_feedback(str(exc)))
                last_reason = "schema_validation"
                continue

            node_count = self._count_nodes(data)
            if node_count < self.options.min_total_nodes:
                corrections.append(self._detail_feedback(node_count, self.options.min_total_nodes))
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

    def _call_openrouter_text_only(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.options.api_key}",
            "Content-Type": "application/json",
        }
        if self.options.referer:
            headers["HTTP-Referer"] = self.options.referer
        if self.options.title:
            headers["X-Title"] = self.options.title

        payload = {
            "model": self.options.model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a meticulous document analyst. Follow the user instructions exactly and answer in JSON only.",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            "temperature": self.options.temperature,
            "max_tokens": self.options.max_tokens,
        }

        if self.options.response_format:
            payload["response_format"] = {"type": self.options.response_format}

        response = requests.post(
            self.options.endpoint,
            headers=headers,
            json=payload,
            timeout=180,
        )
        self._last_debug["status_code"] = response.status_code
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"Unexpected OpenRouter response: {data}")

        message = choices[0].get("message", {})
        raw_text = self._extract_message_text(message)
        if not raw_text:
            raise ValueError(f"Empty response content from OpenRouter: {data}")
        self._last_debug["usage"] = data.get("usage", {})
        return raw_text.strip()
