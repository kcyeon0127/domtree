"""Tree data structures and helpers used across the DOMTree project."""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional

from .schema import NodeMetadata


@dataclasses.dataclass
class TreeNode:
    """Simple general tree node with arbitrary metadata."""

    name: str
    label: Optional[str] = None
    children: List["TreeNode"] = dataclasses.field(default_factory=list)
    attributes: Dict[str, Any] = dataclasses.field(default_factory=dict)
    metadata: NodeMetadata = dataclasses.field(default_factory=lambda: NodeMetadata(node_type="generic"))
    identifier: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

    def traverse(self) -> Iterable["TreeNode"]:
        queue = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def iter_leaves(self) -> Iterable["TreeNode"]:
        for node in self.traverse():
            if not node.children:
                yield node

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "attributes": self.attributes,
            "metadata": self.metadata.to_dict(),
            "identifier": self.identifier,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        metadata = data.get("metadata")
        meta = NodeMetadata.from_dict(metadata) if isinstance(metadata, dict) else NodeMetadata(node_type="generic")
        node = cls(
            name=data["name"],
            label=data.get("label"),
            attributes=data.get("attributes", {}),
            metadata=meta,
            identifier=data.get("identifier", uuid.uuid4().hex),
        )
        for child in data.get("children", []):
            node.add_child(cls.from_dict(child))
        return node

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def pretty_print(self, *, indent: str = "  ", level: int = 0) -> str:
        prefix = indent * level
        label = f" ({self.label})" if self.label else ""
        meta = ""
        if self.metadata:
            meta = " " + json.dumps(self.metadata.to_dict(), ensure_ascii=False)
        line = f"{prefix}- {self.name}{label}{meta}\n"
        for child in self.children:
            line += child.pretty_print(indent=indent, level=level + 1)
        return line

    def map(self, func: Callable[["TreeNode"], "TreeNode"]) -> "TreeNode":
        new_node = func(dataclasses.replace(self, children=[]))
        for child in self.children:
            new_node.add_child(child.map(func))
        return new_node

    def copy(self) -> "TreeNode":
        return TreeNode.from_dict(self.to_dict())


def tree_size(node: TreeNode) -> int:
    return sum(1 for _ in node.traverse())


def tree_depth(node: TreeNode) -> int:
    max_depth = 0
    stack = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in current.children:
            stack.append((child, depth + 1))
    return max_depth
