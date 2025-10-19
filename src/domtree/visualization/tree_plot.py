"""Matplotlib visualisations for DOM trees."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import networkx as nx

from ..tree import TreeNode


ASSET_FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "NanumGothic-Regular.ttf"
ASSET_FONT_NAME = None
if ASSET_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(ASSET_FONT_PATH))
    ASSET_FONT_NAME = font_manager.FontProperties(fname=str(ASSET_FONT_PATH)).get_name()


_FONT_CANDIDATES = [name for name in ([ASSET_FONT_NAME] if ASSET_FONT_NAME else [])]
_FONT_CANDIDATES += [
    "NanumGothic",
    "AppleGothic",
    "Malgun Gothic",
    "Arial Unicode MS",
    "Noto Sans CJK KR",
    "PingFang HK",
    "PingFang SC",
    "PingFang TC",
    "DejaVu Sans",
]


def _configure_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in _FONT_CANDIDATES:
        if family and family in available:
            plt.rcParams["font.family"] = [family]
            plt.rcParams["font.sans-serif"] = [family]
            break
    else:
        plt.rcParams.setdefault("font.family", ["DejaVu Sans"])
    plt.rcParams.setdefault("axes.unicode_minus", False)


_configure_font()


def _build_graph(node: TreeNode) -> nx.DiGraph:
    graph = nx.DiGraph()

    def _add(current: TreeNode):
        graph.add_node(current.identifier, label=current.label or current.name)
        for child in current.children:
            graph.add_edge(current.identifier, child.identifier)
            _add(child)

    _add(node)
    return graph


def _hierarchy_positions(graph: nx.DiGraph, root: str, width: float = 1.0, vert_gap: float = 0.2, vert_loc: float = 0.0) -> dict:
    """Recursively assign positions for a tree graph."""

    def _hierarchy(node: str, width: float, vert_loc: float, x_center: float, pos: dict) -> dict:
        children = list(graph.successors(node))
        if not children:
            pos[node] = (x_center, vert_loc)
            return pos
        dx = width / len(children)
        next_x = x_center - width / 2 + dx / 2
        pos[node] = (x_center, vert_loc)
        for child in children:
            pos = _hierarchy(child, dx, vert_loc - vert_gap, next_x, pos)
            next_x += dx
        return pos

    return _hierarchy(root, width, vert_loc, 0.5, {})


def plot_tree(tree: TreeNode, *, title: str = "Tree", figsize: Tuple[int, int] = (8, 6), path: Path | None = None) -> None:
    graph = _build_graph(tree)
    pos = _hierarchy_positions(graph, tree.identifier)
    labels = nx.get_node_attributes(graph, "label")

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos, node_size=600, node_color="#90caf9", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=False, ax=ax)
    label_kwargs = {"font_size": 8, "ax": ax}
    if ASSET_FONT_NAME is not None:
        label_kwargs["font_family"] = ASSET_FONT_NAME
    nx.draw_networkx_labels(graph, pos, labels, **label_kwargs)
    ax.set_title(title)
    ax.axis("off")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(human_tree: TreeNode, llm_tree: TreeNode, *, path: Path | None = None) -> None:
    graph_left = _build_graph(human_tree)
    graph_right = _build_graph(llm_tree)

    pos_left = _hierarchy_positions(graph_left, human_tree.identifier)
    pos_right = _hierarchy_positions(graph_right, llm_tree.identifier)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, graph, pos, title in zip(
        axes,
        (graph_left, graph_right),
        (pos_left, pos_right),
        ("Human Tree", "LLM Tree"),
    ):
        labels = nx.get_node_attributes(graph, "label")
        nx.draw_networkx_nodes(graph, pos, node_size=600, node_color="#a5d6a7", ax=ax)
        nx.draw_networkx_edges(graph, pos, arrows=False, ax=ax)
        label_kwargs = {"font_size": 8, "ax": ax}
        if ASSET_FONT_NAME is not None:
            label_kwargs["font_family"] = ASSET_FONT_NAME
        nx.draw_networkx_labels(graph, pos, labels, **label_kwargs)
        ax.set_title(title)
        ax.axis("off")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
