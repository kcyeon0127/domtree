"""Heuristics for extracting a human-perceived logical tree from HTML."""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from .schema import NodeMetadata, VisualCues
from .tree import TreeNode

logger = logging.getLogger(__name__)

HEADING_TAGS = {f"h{i}" for i in range(1, 7)}
SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "link",
    "meta",
    "svg",
}
TEXTUAL_BLOCKS = {
    "p",
    "div",
    "article",
    "section",
    "blockquote",
    "li",
    "dd",
    "dt",
}
STRUCTURAL_ZONES = {
    "main": "main",
    "aside": "sidebar",
    "nav": "nav",
    "header": "header",
    "footer": "footer",
    "section": "section",
    "article": "section",
}


@dataclasses.dataclass
class HumanTreeOptions:
    min_text_length: int = 20
    heading_text_limit: int = 200
    text_preview_limit: int = 200
    dominant_language: Optional[str] = None
    include_lists: bool = True
    include_tables: bool = True
    include_figures: bool = True
    restrict_to_viewport: bool = True
    include_text_nodes: bool = False
    max_list_items: int = 5


@dataclasses.dataclass
class HumanTreeBundle:
    zone_tree: TreeNode
    heading_tree: TreeNode
    contraction_tree: TreeNode


class HumanTreeExtractor:
    """Build a logical tree approximating how a human perceives the page layout."""

    def __init__(self, html: str, *, url: Optional[str] = None, options: Optional[HumanTreeOptions] = None):
        self.raw_html = html
        self.url = url
        self.options = options or HumanTreeOptions()
        self.soup = self._prepare_soup(html)
        self._reading_counter = 1
        self._heading_counter = 1
        self._processed: set[int] = set()
        self.viewport_width: Optional[float] = None
        self.viewport_height: Optional[float] = None

    def extract(self) -> HumanTreeBundle:
        body = self.soup.body or self.soup
        title_text = self.soup.title.get_text(strip=True) if self.soup.title else ""
        root_label = self.url or title_text or "document"

        self._processed.clear()
        self._reading_counter = 1
        zone_tree = self._build_zone_tree(body, root_label, title_text)

        self._processed.clear()
        self._heading_counter = 1
        heading_tree = self._build_heading_tree(body, root_label, title_text)

        self._processed.clear()
        self._heading_counter = 1
        contraction_tree = self._build_contraction_tree(body, root_label, title_text)

        return HumanTreeBundle(zone_tree=zone_tree, heading_tree=heading_tree, contraction_tree=contraction_tree)

    def _build_zone_tree(self, body: Tag, root_label: str, title_text: str) -> TreeNode:
        root_meta = NodeMetadata(
            node_type="page",
            role="root",
            reading_order=0,
            text_heading=title_text or None,
            language=self._detect_language(),
        )
        root = TreeNode(name="page", label=root_label, metadata=root_meta)

        zone_elements = list(self._identify_zones(body)) or [body]
        for zone_element in zone_elements:
            zone_node = self._build_zone_node(zone_element)
            if zone_node:
                root.add_child(zone_node)
        return root

    def _build_heading_tree(self, body: Tag, root_label: str, title_text: str) -> TreeNode:
        heading_label = f"{root_label} (headings)" if root_label else "headings"
        root_meta = NodeMetadata(
            node_type="page",
            role="heading_root",
            reading_order=0,
            text_heading=title_text or None,
            language=self._detect_language(),
        )
        root = TreeNode(name="page", label=heading_label, metadata=root_meta)

        stack: List[Tuple[int, TreeNode]] = [(0, root)]
        for heading in body.find_all(HEADING_TAGS):
            if not heading.name or len(heading.name) < 2:
                continue
            level_part = heading.name[1:]
            if not level_part.isdigit():
                continue
            level = int(level_part)
            if not self._is_visible(heading):
                continue
            heading_text = self._normalize_text(heading.get_text(" "))
            metadata = NodeMetadata(
                node_type="section",
                role="section",
                text_heading=heading_text[: self.options.heading_text_limit] if heading_text else None,
                heading_level=level,
                reading_order=self._next_heading_order(),
                dom_refs=[self._dom_ref(heading)],
                visual_cues=self._visual_cues(heading),
            )
            if self.options.restrict_to_viewport and not self._bbox_in_viewport(metadata.visual_cues.bbox):
                continue
            label = metadata.text_heading or heading.name.upper()
            node = TreeNode(name="section", label=label, metadata=metadata)
            while stack and stack[-1][0] >= level:
                stack.pop()
                    parent = stack[-1][1] if stack else root
                    parent.add_child(node)
                    stack.append((level, node))
                return root
            
            def _build_contraction_tree(self, body: Tag, root_label: str, title_text: str) -> TreeNode:
                """Build a tree based on heading tags and their visual prominence."""
                heading_label = f"{root_label} (contraction)" if root_label else "contraction"
                root_meta = NodeMetadata(
                    node_type="page",
                    role="contraction_root",
                    reading_order=0,
                    text_heading=title_text or None,
                    language=self._detect_language(),
                )
                root = TreeNode(name="page", label=heading_label, metadata=root_meta)
            
                headings = []
                for element in body.find_all(list(HEADING_TAGS) + ["p", "div"]):
                    prominence = self._get_heading_prominence(element)
                    if prominence > 0:
                        headings.append((prominence, element))
            
                # Sort by document order, assuming prominence is stable
                headings.sort(key=lambda x: x[1].sourceline or 0)
            
                stack: List[Tuple[float, TreeNode]] = [(float("inf"), root)]
            
                for prominence, element in headings:
                    if not self._is_visible(element):
                        continue
            
                    heading_text = self._normalize_text(element.get_text(" "))
                    level = int(element.name[1]) if element.name in HEADING_TAGS else None
            
                    metadata = NodeMetadata(
                        node_type="section",
                        role="section",
                        text_heading=heading_text[: self.options.heading_text_limit] if heading_text else None,
                        heading_level=level,
                        reading_order=self._next_heading_order(),
                        dom_refs=[self._dom_ref(element)],
                        visual_cues=self._visual_cues(element),
                        notes={"prominence_score": prominence},
                    )
                    if self.options.restrict_to_viewport and not self._bbox_in_viewport(metadata.visual_cues.bbox):
                        continue
            
                    label = metadata.text_heading or element.name.upper()
                    node = TreeNode(name="section", label=label, metadata=metadata)
            
                    while stack and stack[-1][0] <= prominence:
                        stack.pop()
            
                    parent = stack[-1][1] if stack else root
                    parent.add_child(node)
                    stack.append((prominence, node))
            
                return root
            
            
            def _get_heading_prominence(self, element: Tag) -> float:
                """Calculate a prominence score for an element to see if it acts as a heading."""
                if not isinstance(element, Tag):
                    return 0
            
                score = 0.0
                tag_name = element.name
                if tag_name in HEADING_TAGS:
                    level = int(tag_name[1])
                    score += (7 - level) * 10  # H1=60, H2=50, ..., H6=10
            
                aria_level = element.get("aria-level")
                if element.get("role") == "heading" and aria_level and aria_level.isdigit():
                    level = int(aria_level)
                    score = max(score, (7 - level) * 10)
            
                cues = self._visual_cues(element)
                if cues.font_size:
                    score += cues.font_size
            
                if cues.font_weight:
                    try:
                        weight_val = int(cues.font_weight)
                        if weight_val >= 600:
                            score += 10
                    except ValueError:
                        if cues.font_weight == "bold":
                            score += 10
            
                # Not a heading if it doesn't have a base score from tag/role and is small
                if score < 10 and not self._normalize_text(element.get_text(" ")):
                    return 0
            
                return score
    # --------------------------- zone detection ----------------------------

    def _identify_zones(self, body: Tag) -> Iterable[Tag]:
        primary = [child for child in body.find_all(recursive=False) if isinstance(child, Tag)]
        if not primary:
            return []
        zones: List[Tag] = []
        for candidate in primary:
            if candidate.name in SKIP_TAGS:
                continue
            if not self._is_visible(candidate):
                continue
            if self.options.restrict_to_viewport and not self._element_in_viewport(candidate):
                continue
            zones.append(candidate)
        if not zones:
            zones = primary
        return zones

    def _build_zone_node(self, element: Tag) -> Optional[TreeNode]:
        role = self._classify_zone_role(element)
        heading_text = self._first_heading_text(element)
        metadata = NodeMetadata(
            node_type="zone",
            role=role,
            text_heading=heading_text,
            heading_level=None,
            reading_order=self._next_order(),
            dom_refs=[self._dom_ref(element)],
            visual_cues=self._visual_cues(element),
        )
        label = heading_text or role or element.name
        if self.options.restrict_to_viewport and not self._bbox_in_viewport(metadata.visual_cues.bbox):
            return None
        zone_node = TreeNode(name="zone", label=label, metadata=metadata)
        self._processed.add(id(element))

        heading_nodes = self._build_heading_hierarchy(element, zone_node)
        if not heading_nodes:
            for content_node in self._collect_zone_content(element, metadata):
                zone_node.add_child(content_node)
        return zone_node

    def _classify_zone_role(self, element: Tag) -> str:
        if element.name in STRUCTURAL_ZONES:
            return STRUCTURAL_ZONES[element.name]
        classes = " ".join(element.get("class", [])).lower()
        if "sidebar" in classes:
            return "sidebar"
        if "toc" in classes or "table-of-contents" in classes:
            return "toc"
        if "footer" in classes:
            return "footer"
        if "header" in classes or "top" in classes:
            return "header"
        if "nav" in classes or "menu" in classes:
            return "nav"
        if "ad" in classes or "banner" in classes:
            return "ad"
        return "main"

    # --------------------------- headings & content ------------------------

    def _build_heading_hierarchy(self, zone_element: Tag, zone_node: TreeNode) -> List[TreeNode]:
        headings: List[Tuple[Tag, int]] = []
        for heading in zone_element.find_all(HEADING_TAGS):
            level = int(heading.name[1])
            headings.append((heading, level))
            self._processed.add(id(heading))
        if not headings:
            return []

        stack: List[Tuple[int, TreeNode]] = [(0, zone_node)]
        created_nodes: List[TreeNode] = []
        for idx, (heading, level) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_node = stack[-1][1] if stack else zone_node
            section_node = self._create_section_node(heading, level)
            if section_node is None:
                continue
            parent_node.add_child(section_node)
            stack.append((level, section_node))
            created_nodes.append(section_node)

            next_heading = headings[idx + 1][0] if idx + 1 < len(headings) else None
            for content_node in self._collect_content_between(heading, next_heading, zone_element):
                section_node.add_child(content_node)
        return created_nodes

    def _create_section_node(self, heading: Tag, level: int) -> TreeNode:
        heading_text = self._normalize_text(heading.get_text(" "))
        metadata = NodeMetadata(
            node_type="section",
            role="section",
            text_heading=heading_text[: self.options.heading_text_limit] if heading_text else None,
            heading_level=level,
            reading_order=self._next_order(),
            dom_refs=[self._dom_ref(heading)],
            visual_cues=self._visual_cues(heading),
        )
        if self.options.restrict_to_viewport and not self._bbox_in_viewport(metadata.visual_cues.bbox):
            return None
        node = TreeNode(name="section", label=metadata.text_heading or f"H{level}", metadata=metadata)
        return node

    def _collect_content_between(self, start: Tag, end: Optional[Tag], zone: Tag) -> Iterable[TreeNode]:
        nodes: List[TreeNode] = []
        current = start.next_element
        while current and current is not end:
            if isinstance(current, Tag):
                if id(current) in self._processed:
                    current = current.next_element
                    continue
                if current.name in SKIP_TAGS:
                    current = current.next_element
                    continue
                if not self._is_descendant(current, zone):
                    break
                content_node = self._collect_content_node(current)
                if content_node:
                    nodes.append(content_node)
                    self._consume_subtree(current)
                current = current.next_element
            elif isinstance(current, NavigableString):
                if self.options.include_text_nodes:
                    text = self._normalize_text(str(current))
                    if text and len(text) >= self.options.min_text_length:
                        text_node = self._create_text_node(text)
                        nodes.append(text_node)
                current = current.next_element
            else:
                current = current.next_element
        return nodes

    def _collect_zone_content(self, element: Tag, parent_metadata: NodeMetadata) -> Iterable[TreeNode]:
        nodes: List[TreeNode] = []
        for child in element.children:
            if isinstance(child, Tag):
                if child.name in SKIP_TAGS or id(child) in self._processed:
                    continue
                node = self._collect_content_node(child, parent_metadata=parent_metadata)
                if node:
                    nodes.append(node)
                    self._consume_subtree(child)
            elif isinstance(child, NavigableString):
                if self.options.include_text_nodes:
                    text = self._normalize_text(str(child))
                    if text and len(text) >= self.options.min_text_length:
                        nodes.append(self._create_text_node(text))
        return nodes

    def _collect_content_node(self, element: Tag, parent_metadata: Optional[NodeMetadata] = None) -> Optional[TreeNode]:
        node_type = self._infer_content_type(element)
        if node_type is None:
            return None
        text_preview = self._extract_text_preview(element)
        metadata = NodeMetadata(
            node_type=node_type,
            role=(parent_metadata.role if parent_metadata else None),
            reading_order=self._next_order(),
            dom_refs=[self._dom_ref(element)],
            visual_cues=self._visual_cues(element),
            text_preview=text_preview,
        )
        if self.options.restrict_to_viewport and not self._bbox_in_viewport(metadata.visual_cues.bbox):
            return None
        label = text_preview or element.name
        node = TreeNode(name=node_type, label=label, metadata=metadata)
        self._processed.add(id(element))

        if element.name in {"ul", "ol"} and self.options.include_lists:
            for li in element.find_all("li", recursive=False):
                li_text = self._extract_text_preview(li)
                li_meta = NodeMetadata(
                    node_type="list_item",
                    role="list_item",
                    reading_order=self._next_order(),
                    dom_refs=[self._dom_ref(li)],
                    text_preview=li_text,
                    visual_cues=self._visual_cues(li),
                )
                if self.options.restrict_to_viewport and not self._bbox_in_viewport(li_meta.visual_cues.bbox):
                    continue
                node.add_child(TreeNode(name="list_item", label=li_text or "item", metadata=li_meta))
                self._processed.add(id(li))
                if len(node.children) >= self.options.max_list_items:
                    break
        return node

    def _create_text_node(self, text: str) -> TreeNode:
        metadata = NodeMetadata(
            node_type="paragraph",
            role="body",
            reading_order=self._next_order(),
            text_preview=text[: self.options.text_preview_limit],
            visual_cues=VisualCues(),
        )
        return TreeNode(name="paragraph", label=metadata.text_preview or text, metadata=metadata)

    # --------------------------- helpers -----------------------------------

    def _detect_language(self) -> Optional[str]:
        html_tag = self.soup.find("html")
        if html_tag and html_tag.has_attr("lang"):
            return html_tag["lang"].split("-")[0]
        return self.options.dominant_language

    def _prepare_soup(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(SKIP_TAGS):
            tag.decompose()
        self._extract_viewport_meta(soup)
        return soup

    def _extract_viewport_meta(self, soup: BeautifulSoup) -> None:
        body = soup.body
        if not body:
            return
        attr = body.get("data-domtree-viewport")
        if not attr:
            return
        try:
            width_str, height_str = attr.split(",")
            self.viewport_width = float(width_str)
            self.viewport_height = float(height_str)
        except (ValueError, TypeError):
            self.viewport_width = None
            self.viewport_height = None

    def _is_visible(self, element: Tag) -> bool:
        classes = element.get("class", [])
        if any(cls for cls in classes if "hidden" in cls or "sr-only" in cls):
            return False
        style = element.get("style", "")
        if "display:none" in style.replace(" ","").lower():
            return False
        return True

    def _first_heading_text(self, element: Tag) -> Optional[str]:
        heading = element.find(HEADING_TAGS)
        if heading:
            return self._normalize_text(heading.get_text(" "))[: self.options.heading_text_limit]
        return None

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _dom_ref(self, element: Tag) -> str:
        if element.has_attr("id"):
            return f"#{element['id']}"
        classes = element.get("class", [])
        class_selector = "".join(f".{cls}" for cls in classes[:3])
        return f"{element.name}{class_selector}" if class_selector else element.name

    def _visual_cues(self, element: Tag) -> VisualCues:
        style = element.get("style", "")
        parsed = self._parse_styles(style)
        font_weight = None
        if element.name in HEADING_TAGS:
            font_weight = "bold"
        elif "font-weight" in parsed:
            font_weight = parsed["font-weight"]
        font_size_raw = parsed.get("font-size")
        font_size = self._parse_css_size(font_size_raw) if font_size_raw else None
        margin_top = self._parse_css_size(parsed.get("margin-top"))
        margin_bottom = self._parse_css_size(parsed.get("margin-bottom"))
        bg_color = parsed.get("background") or parsed.get("background-color")
        bbox = self._extract_bbox(element)
        return VisualCues(
            font_size=font_size,
            font_weight=font_weight,
            margin_top=margin_top,
            margin_bottom=margin_bottom,
            bg_color=bg_color,
            bbox=bbox,
        )

    def _extract_bbox(self, element: Tag) -> Optional[Tuple[float, float, float, float]]:
        attr = element.get("data-domtree-bbox")
        if not attr:
            return None
        try:
            top, left, bottom, right = (float(value) for value in attr.split(","))
            return (top, left, bottom, right)
        except (ValueError, TypeError):
            return None

    def _bbox_in_viewport(self, bbox: Optional[Tuple[float, float, float, float]]) -> bool:
        if not self.options.restrict_to_viewport:
            return True
        if bbox is None or self.viewport_height is None:
            return True
        top, _, bottom, _ = bbox
        return bottom > 0 and top < self.viewport_height

    def _element_in_viewport(self, element: Tag) -> bool:
        bbox = self._extract_bbox(element)
        return self._bbox_in_viewport(bbox)

    def _parse_styles(self, style: str) -> dict:
        result = {}
        for part in style.split(";"):
            if ":" in part:
                key, value = part.split(":", 1)
                result[key.strip().lower()] = value.strip()
        return result

    def _parse_css_size(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        match = re.match(r"([0-9.]+)", value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _infer_content_type(self, element: Tag) -> Optional[str]:
        if element.name in HEADING_TAGS:
            return None
        if element.name in {"p", "div", "span"}:
            text = self._extract_text_preview(element)
            if not text or len(text) < self.options.min_text_length:
                return None
            return "paragraph"
        if element.name in {"ul", "ol"} and self.options.include_lists:
            return "list"
        if element.name == "table" and self.options.include_tables:
            return "table"
        if element.name in {"figure", "img", "picture"} and self.options.include_figures:
            return "figure"
        if element.name == "blockquote":
            return "quote"
        if element.name in {"code", "pre"}:
            return "code"
        return None

    def _extract_text_preview(self, element: Tag) -> Optional[str]:
        text = self._normalize_text(element.get_text(" "))
        if not text:
            return None
        return text[: self.options.text_preview_limit]

    def _consume_subtree(self, element: Tag) -> None:
        for descendant in element.descendants:
            if isinstance(descendant, Tag):
                self._processed.add(id(descendant))

    def _is_descendant(self, element: Tag, ancestor: Tag) -> bool:
        return ancestor in element.parents

    def _next_order(self) -> int:
        value = self._reading_counter
        self._reading_counter += 1
        return value

    def _next_heading_order(self) -> int:
        value = self._heading_counter
        self._heading_counter += 1
        return value


def extract_human_tree(html: str, *, url: Optional[str] = None, options: Optional[HumanTreeOptions] = None) -> HumanTreeBundle:
    extractor = HumanTreeExtractor(html, url=url, options=options)
    return extractor.extract()
