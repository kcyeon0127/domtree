"""Canonical categories used to compare human DOM trees with LLM trees.

The base schema interprets labels from the following layout-analysis datasets:

* **PubLayNet (Zhong et al., 2019, arXiv:1908.07836)** – provides page-level
  annotations for ``Title``, ``Text``, ``Figure``, ``Table``, ``List``.
* **DocLayNet (Pfitzmann et al., 2022, arXiv:2210.05068)** – extends the set with
  ``Caption``, ``Footnote``, ``Page-footer`` 등 추가 영역.

웹페이지는 PDF 문서와 구조가 다르므로, 두 데이터셋의 공통 상위 개념을
웹 요소 역할(`zone_main`, `section`, `list_item` 등)과 매핑하도록 정리했다.
각 카테고리는 ``CATEGORY_DEFINITIONS`` 에 설명 문자열을 포함하고,
``CATEGORY_ALIASES`` 는 DOM/LLM 트리에 등장하는 이름을 표준 카테고리에
맞춰 정규화하기 위한 사전이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class Category:
    """Simple container describing a canonical layout category."""

    name: str
    description: str
    source: str


# Canonical category list (order is not important but kept stable for reports)
CATEGORY_DEFINITIONS: Dict[str, Category] = {
    "TITLE": Category(
        name="TITLE",
        description="Global document title or primary page heading.",
        source="PubLayNet/DocLayNet",
    ),
    "SECTION_HEADING": Category(
        name="SECTION_HEADING",
        description="Sub-headings (H1-H6, nav group labels, sidebar titles).",
        source="DocLayNet",
    ),
    "BODY_TEXT": Category(
        name="BODY_TEXT",
        description="Main prose paragraphs within the article or content body.",
        source="PubLayNet",
    ),
    "SIDEBAR": Category(
        name="SIDEBAR",
        description="Complementary content such as aside panels, callouts, widgets.",
        source="Web-specific extension",
    ),
    "NAVIGATION": Category(
        name="NAVIGATION",
        description="Menus, table-of-contents, pagination, header nav bars.",
        source="Web-specific extension",
    ),
    "LIST": Category(
        name="LIST",
        description="Ordered/unordered lists that enumerate items in the body.",
        source="PubLayNet",
    ),
    "TABLE": Category(
        name="TABLE",
        description="Tabular data, grid layouts, property/value tables.",
        source="PubLayNet",
    ),
    "FIGURE": Category(
        name="FIGURE",
        description="Images, charts, infographics embedded in the content.",
        source="PubLayNet",
    ),
    "CAPTION": Category(
        name="CAPTION",
        description="Captions or labels describing figures/tables.",
        source="DocLayNet",
    ),
    "FOOTNOTE": Category(
        name="FOOTNOTE",
        description="Footnotes, disclaimers, copyright or footer text.",
        source="DocLayNet",
    ),
    "ADVERT": Category(
        name="ADVERT",
        description="Advertisement or sponsored blocks detectable by role/class.",
        source="Web-specific extension",
    ),
}


# Map DOM/LLM node hints -> canonical category.
# Keys are normalized lowercase tokens, values are canonical category keys.
CATEGORY_ALIASES: Dict[str, str] = {
    # Titles / headings
    "page": "TITLE",
    "title": "TITLE",
    "zone_header": "SECTION_HEADING",
    "section_header": "SECTION_HEADING",
    "section_title": "SECTION_HEADING",
    "heading": "SECTION_HEADING",
    "toc": "NAVIGATION",
    "table-of-contents": "NAVIGATION",
    "nav": "NAVIGATION",
    "menu": "NAVIGATION",
    "breadcrumb": "NAVIGATION",
    # Body / sidebar / footnote
    "zone_main": "BODY_TEXT",
    "paragraph": "BODY_TEXT",
    "text": "BODY_TEXT",
    "article": "BODY_TEXT",
    "section": "BODY_TEXT",
    "aside": "SIDEBAR",
    "sidebar": "SIDEBAR",
    "footnote": "FOOTNOTE",
    "footer": "FOOTNOTE",
    # Media and structured content
    "list": "LIST",
    "list_item": "LIST",
    "table": "TABLE",
    "table_row": "TABLE",
    "figure": "FIGURE",
    "image": "FIGURE",
    "img": "FIGURE",
    "caption": "CAPTION",
    "figcaption": "CAPTION",
    "table_caption": "CAPTION",
    # Misc
    "ad": "ADVERT",
    "advert": "ADVERT",
    "sponsored": "ADVERT",
}


def canonical_category(name: str) -> str | None:
    """Return the canonical category key for a raw node name.

    Parameters
    ----------
    name:
        Raw ``TreeNode.name`` or ``metadata.role``/``metadata.type`` string.
    """

    normalized = name.strip().lower()
    return CATEGORY_ALIASES.get(normalized)


def known_categories() -> Iterable[str]:
    """Convenience accessor returning all canonical category names."""

    return CATEGORY_DEFINITIONS.keys()
