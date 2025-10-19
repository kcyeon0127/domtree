"""Heuristics for extracting a human-perceived logical tree from HTML."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import Counter
from typing import Iterable, List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

from .tree import TreeNode

logger = logging.getLogger(__name__)

HEADING_TAGS = {f"h{i}" for i in range(1, 7)}
BLOCK_TAGS = {
    "article",
    "section",
    "nav",
    "aside",
    "header",
    "footer",
    "main",
    "div",
    "ul",
    "ol",
    "li",
    "p",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "figure",
    "figcaption",
    "form",
    "fieldset",
    "legend",
}
SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "link",
    "meta",
    "svg",
    "path",
    "circle",
    "rect",
}
TEXTUAL_TAGS = {
    "p",
    "li",
    "span",
    "strong",
    "em",
    "blockquote",
    "code",
    "pre",
    "label",
}
INTERACTIVE_TAGS = {"button", "a"}
SEMANTIC_KEYWORDS = {
    "hero",
    "header",
    "nav",
    "menu",
    "footer",
    "sidebar",
    "content",
    "main",
    "article",
    "card",
    "section",
    "cta",
    "gallery",
    "grid",
    "list",
    "table",
    "form",
}


@dataclasses.dataclass
class HumanTreeOptions:
    min_text_length: int = 25
    max_depth: int = 8
    heading_text_limit: int = 200
    text_preview_limit: int = 160
    merge_short_text_nodes: bool = True
    preserve_lists: bool = True
    include_tables: bool = True
    density_threshold: float = 0.08
    dominant_language: Optional[str] = None


class HumanTreeExtractor:
    """Build a logical tree approximating how a human perceives the page layout."""

    def __init__(self, html: str, *, url: Optional[str] = None, options: Optional[HumanTreeOptions] = None):
        self.raw_html = html
        self.url = url
        self.options = options or HumanTreeOptions()
        self.soup = self._prepare_soup(html)

    def extract(self) -> TreeNode:
        body = self.soup.body or self.soup
        root_label = self.url or (self.soup.title.string.strip() if self.soup.title and self.soup.title.string else "document")
        root = TreeNode(
            name="document",
            label=root_label,
            metadata={
                "language": self._detect_language(),
                "title": root_label,
            },
        )

        for child in body.children:
            node = self._element_to_node(child, depth=1)
            if node:
                root.add_child(node)

        self._postprocess(root)
        return root

    def _prepare_soup(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(SKIP_TAGS):
            tag.decompose()
        return soup

    def _detect_language(self) -> Optional[str]:
        html_tag = self.soup.find("html")
        if html_tag and html_tag.has_attr("lang"):
            return html_tag["lang"].split("-")[0]
        return self.options.dominant_language

    def _element_to_node(self, element, depth: int) -> Optional[TreeNode]:
        if isinstance(element, NavigableString):
            text = self._normalize_text(str(element))
            if not text:
                return None
            return TreeNode(
                name="text",
                label=text[: self.options.text_preview_limit],
                metadata={
                    "text_length": len(text),
                    "density": 1.0,
                    "kind": "text",
                },
            )

        if not isinstance(element, Tag):
            return None

        if element.name in SKIP_TAGS:
            return None

        text_fragments: List[str] = []
        children: List[TreeNode] = []

        for child in element.children:
            if isinstance(child, NavigableString):
                text = self._normalize_text(str(child))
                if text:
                    text_fragments.append(text)
            else:
                converted = self._element_to_node(child, depth=depth + 1)
                if converted:
                    children.append(converted)

        text_content = " ".join(text_fragments).strip()
        text_length = len(text_content)
        descendant_tags = max(1, len(list(element.descendants)))
        density = text_length / descendant_tags

        role = self._classify_role(element)
        heading = self._find_heading(element)
        include_current = self._should_include(element, text_length=text_length, density=density, role=role, depth=depth)

        if not include_current and children:
            # Promote children upward unless we want to keep container to preserve grouping (e.g. lists)
            if element.name in {"ul", "ol"} and self.options.preserve_lists:
                include_current = True
            elif element.name == "table" and self.options.include_tables:
                include_current = True
            else:
                return self._merge_children(children, text_content)

        if not include_current and not children:
            if text_length >= self.options.min_text_length and density >= self.options.density_threshold:
                include_current = True

        if not include_current:
            return None

        label = self._derive_label(element, heading=heading, role=role, text_content=text_content)
        metadata = {
            "tag": element.name,
            "classes": list(element.get("class", [])),
            "role": role,
            "heading_level": self._heading_level(heading) if heading else None,
            "text_length": text_length,
            "density": round(density, 3),
            "id": element.get("id"),
            "depth": depth,
        }
        if text_content:
            metadata["text_preview"] = text_content[: self.options.text_preview_limit]
        if heading:
            metadata["heading_text"] = self._normalize_text(heading.get_text(" "))[: self.options.heading_text_limit]
        if element.has_attr("style"):
            metadata["style"] = element["style"]

        node = TreeNode(
            name=role or element.name,
            label=label,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

        for child in children:
            node.add_child(child)

        return node

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _classify_role(self, element: Tag) -> str:
        if element.has_attr("role"):
            return element["role"]
        tag = element.name
        classes = element.get("class", [])
        class_counter = Counter(classes)
        joined = " ".join(classes).lower()
        for keyword in SEMANTIC_KEYWORDS:
            if keyword in joined:
                return keyword
        if tag in {
            "header",
            "footer",
            "main",
            "nav",
            "aside",
            "article",
            "section",
            "form",
            "table",
        }:
            return tag
        if tag == "div":
            if "hero" in joined:
                return "hero"
            if any(cls.endswith("-wrapper") for cls in classes):
                return "wrapper"
            if "sidebar" in joined:
                return "sidebar"
        if tag in INTERACTIVE_TAGS:
            return "interactive"
        if tag in TEXTUAL_TAGS:
            return "text"
        return tag

    def _should_include(self, element: Tag, *, text_length: int, density: float, role: str, depth: int) -> bool:
        if depth > self.options.max_depth:
            return False
        if element.name in HEADING_TAGS:
            return True
        if role in {"nav", "header", "footer", "hero", "sidebar"}:
            return True
        if element.name in {"section", "article", "main", "form"}:
            return True
        if element.name in {"ul", "ol", "li"} and self.options.preserve_lists:
            return True
        if element.name == "table" and self.options.include_tables:
            return True
        if text_length >= self.options.min_text_length and density >= self.options.density_threshold:
            return True
        if element.name in TEXTUAL_TAGS and text_length > 10:
            return True
        if element.name in INTERACTIVE_TAGS and text_length > 0:
            return True
        return False

    def _find_heading(self, element: Tag) -> Optional[Tag]:
        if element.name in HEADING_TAGS:
            return element
        for child in element.children:
            if isinstance(child, Tag) and child.name in HEADING_TAGS:
                return child
        return None

    def _heading_level(self, heading: Optional[Tag]) -> Optional[int]:
        if not heading:
            return None
        if heading.name in HEADING_TAGS:
            return int(heading.name[1])
        return None

    def _derive_label(self, element: Tag, *, heading: Optional[Tag], role: str, text_content: str) -> str:
        if heading:
            return self._normalize_text(heading.get_text(" "))[: self.options.heading_text_limit]
        if element.has_attr("aria-label"):
            return element["aria-label"]
        if element.has_attr("alt"):
            return element["alt"]
        if element.name in {"button", "a"} and text_content:
            return text_content[: self.options.text_preview_limit]
        if role and role != element.name:
            return role
        if text_content:
            return text_content[: self.options.text_preview_limit]
        if element.has_attr("id"):
            return element["id"]
        classes = element.get("class", [])
        if classes:
            return " ".join(classes)[: self.options.text_preview_limit]
        return element.name

    def _merge_children(self, children: List[TreeNode], text_content: str) -> Optional[TreeNode]:
        if not children and not text_content:
            return None
        if len(children) == 1 and not text_content:
            return children[0]
        node = TreeNode(name="group", label=text_content[: self.options.text_preview_limit] if text_content else None)
        for child in children:
            node.add_child(child)
        if text_content:
            node.metadata["text_preview"] = text_content[: self.options.text_preview_limit]
        return node

    def _postprocess(self, root: TreeNode) -> None:
        """Optional clean-ups like collapsing consecutive text nodes."""

        if not self.options.merge_short_text_nodes:
            return

        def _merge(node: TreeNode) -> TreeNode:
            merged_children: List[TreeNode] = []
            buffer_text = []
            for child in node.children:
                if child.name == "text" and child.metadata.get("text_length", 0) < self.options.min_text_length:
                    buffer_text.append(child.metadata.get("text_preview") or child.label or "")
                    continue
                if buffer_text:
                    preview = " ".join(buffer_text)
                    merged_children.append(
                        TreeNode(
                            name="text",
                            label=preview[: self.options.text_preview_limit],
                            metadata={"text_length": len(preview), "kind": "text"},
                        )
                    )
                    buffer_text.clear()
                merged_children.append(child)
            if buffer_text:
                preview = " ".join(buffer_text)
                merged_children.append(
                    TreeNode(
                        name="text",
                        label=preview[: self.options.text_preview_limit],
                        metadata={"text_length": len(preview), "kind": "text"},
                    )
                )
            node.children = [ _merge(child) for child in merged_children ]
            return node

        _merge(root)


def extract_human_tree(html: str, *, url: Optional[str] = None, options: Optional[HumanTreeOptions] = None) -> TreeNode:
    extractor = HumanTreeExtractor(html, url=url, options=options)
    return extractor.extract()
