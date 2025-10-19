"""Tree edit distance wrappers built on top of :mod:`zss`."""

from __future__ import annotations

from typing import Callable

from zss import Node, simple_distance

from ..tree import TreeNode, tree_size


def _node_label(node: TreeNode, *, use_label: bool = True) -> str:
    candidate = (node.label or "") if use_label else ""
    heading = node.metadata.text_heading if hasattr(node.metadata, "text_heading") else None
    return candidate or heading or node.name


def _tree_to_zss(node: TreeNode, *, use_label: bool = True) -> Node:
    znode = Node(_node_label(node, use_label=use_label))
    znode.original = node  # Attach pointer for richer cost functions
    for child in node.children:
        znode.addkid(_tree_to_zss(child, use_label=use_label))
    return znode


def tree_edit_distance(
    reference: TreeNode,
    candidate: TreeNode,
    *,
    insert_cost: Callable[[TreeNode], float] | None = None,
    delete_cost: Callable[[TreeNode], float] | None = None,
    relabel_cost: Callable[[TreeNode, TreeNode], float] | None = None,
    use_label: bool = True,
) -> float:
    """Compute the tree edit distance between two :class:`TreeNode` objects."""

    insert_cost = insert_cost or (lambda node: 1.0)
    delete_cost = delete_cost or (lambda node: 1.0)
    relabel_cost = relabel_cost or (lambda a, b: 0.0 if _node_label(a, use_label=use_label) == _node_label(b, use_label=use_label) else 1.0)

    ref = _tree_to_zss(reference, use_label=use_label)
    cand = _tree_to_zss(candidate, use_label=use_label)

    def _get_children(znode: Node):
        return znode.children

    def _insert_cost(znode: Node) -> float:
        return insert_cost(znode.original)

    def _remove_cost(znode: Node) -> float:
        return delete_cost(znode.original)

    def _update_cost(znode_a: Node, znode_b: Node) -> float:
        return relabel_cost(znode_a.original, znode_b.original)

    try:
        return simple_distance(
            ref,
            cand,
            get_children=_get_children,
            insert_cost=_insert_cost,
            remove_cost=_remove_cost,
            update_cost=_update_cost,
        )
    except TypeError:
        # Older versions of zss expose a simpler signature (no custom costs).
        return simple_distance(ref, cand, get_children=_get_children)


def normalized_tree_edit_distance(reference: TreeNode, candidate: TreeNode, **kwargs) -> float:
    """Return TED normalized by the larger tree size."""

    raw = tree_edit_distance(reference, candidate, **kwargs)
    denom = max(tree_size(reference), tree_size(candidate)) or 1
    return raw / denom
