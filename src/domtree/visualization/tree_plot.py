"""Matplotlib visualisations for DOM trees."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import networkx as nx

from ..tree import TreeNode


ASSET_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
ADDED_FONT_NAMES: list[str] = []
if ASSET_FONT_DIR.exists():
    for font_path in ASSET_FONT_DIR.rglob("*"):
        if font_path.is_file() and font_path.suffix.lower() in FONT_EXTENSIONS:
            try:
                font_manager.fontManager.addfont(str(font_path))
                name = font_manager.FontProperties(fname=str(font_path)).get_name()
                if name not in ADDED_FONT_NAMES:
                    ADDED_FONT_NAMES.append(name)
            except Exception:
                continue


_FONT_CANDIDATES = [
    "Noto Sans CJK KR",
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "Nanum Gothic",
    "AppleGothic",
    "Malgun Gothic",
    "PingFang HK",
    "PingFang SC",
    "PingFang TC",
    "DejaVu Sans",
]
for name in reversed(ADDED_FONT_NAMES):
    if name not in _FONT_CANDIDATES:
        _FONT_CANDIDATES.insert(0, name)


def _configure_font() -> list[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    font_stack: list[str] = []
    for family in _FONT_CANDIDATES:
        if family and family in available and family not in font_stack:
            font_stack.append(family)
    if not font_stack:
        font_stack = ["DejaVu Sans"]
    plt.rcParams["font.family"] = font_stack
    plt.rcParams["font.sans-serif"] = font_stack
    plt.rcParams.setdefault("axes.unicode_minus", False)
    return font_stack


_FONT_STACK = _configure_font()


def _build_graph(node: TreeNode) -> nx.DiGraph:
    graph = nx.DiGraph()

    def _add(current: TreeNode):
        graph.add_node(current.identifier, label=current.label or current.name)
        for child in current.children:
            graph.add_edge(current.identifier, child.identifier)
            _add(child)

    _add(node)
    return graph


def _sanitize_label(text: str | None) -> str:
    if not text:
        return ""
    allowed = []
    for char in text:
        code = ord(char)
        if 32 <= code <= 126:
            allowed.append(char)
        elif 0x1100 <= code <= 0x11FF:  # Hangul Jamo
            allowed.append(char)
        elif 0x3130 <= code <= 0x318F:  # Hangul compatibility jamo
            allowed.append(char)
        elif 0xAC00 <= code <= 0xD7A3:  # Hangul syllables
            allowed.append(char)
    sanitized = "".join(allowed).strip()
    return sanitized or "…"


def _hierarchy_positions(graph: nx.DiGraph, root: str, horiz_gap: float = 0.3, vert_gap: float = 0.2, x_loc: float = 0.0) -> dict:
    """Assign positions so the tree grows from left to right."""

    def _hierarchy(node: str, x_pos: float, y_pos: float, pos: dict) -> dict:
        children = list(graph.successors(node))
        pos[node] = (x_pos, y_pos)
        if not children:
            return pos
        offsets = _child_offsets(len(children), vert_gap)
        for offset, child in zip(offsets, children):
            pos = _hierarchy(child, x_pos + horiz_gap, y_pos + offset, pos)
        return pos

    return _hierarchy(root, x_loc, 0.0, {})


def _child_offsets(count: int, gap: float) -> list[float]:
    if count == 1:
        return [0.0]
    total_span = gap * (count - 1)
    start = -total_span / 2
    return [start + i * gap for i in range(count)]


def plot_tree(
    tree: TreeNode,
    *,
    title: str = "Tree",
    figsize: Tuple[int, int] = (12, 6),
    node_size: int = 800,
    arrows: bool = False,
    font_size: int = 9,
    path: Path | None = None,
) -> None:
    graph = _build_graph(tree)
    pos = _hierarchy_positions(graph, tree.identifier)
    raw_labels = nx.get_node_attributes(graph, "label")
    labels = {node_id: _sanitize_label(label) for node_id, label in raw_labels.items()}

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="#90caf9", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=arrows, arrowstyle="-|>", arrowsize=10, ax=ax)
    label_kwargs = {"font_size": font_size, "ax": ax}
    if _FONT_STACK:
        label_kwargs["font_family"] = _FONT_STACK
    nx.draw_networkx_labels(graph, pos, labels, **label_kwargs)
    ax.set_title(title)
    ax.axis("off")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(
    human_tree: TreeNode,
    llm_tree: TreeNode,
    *,
    figsize: Tuple[int, int] = (18, 7),
    node_size: int = 800,
    font_size: int = 9,
    path: Path | None = None,
) -> None:
    graph_left = _build_graph(human_tree)
    graph_right = _build_graph(llm_tree)

    pos_left = _hierarchy_positions(graph_left, human_tree.identifier)
    pos_right = _hierarchy_positions(graph_right, llm_tree.identifier)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, graph, pos, title in zip(
        axes,
        (graph_left, graph_right),
        (pos_left, pos_right),
        ("Human Tree", "LLM Tree"),
    ):
        raw_labels = nx.get_node_attributes(graph, "label")
        labels = {node_id: _sanitize_label(label) for node_id, label in raw_labels.items()}
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="#a5d6a7", ax=ax)
        nx.draw_networkx_edges(graph, pos, arrows=False, ax=ax)
        label_kwargs = {"font_size": font_size, "ax": ax}
        if _FONT_STACK:
            label_kwargs["font_family"] = _FONT_STACK
        nx.draw_networkx_labels(graph, pos, labels, **label_kwargs)
        ax.set_title(title)
        ax.axis("off")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
