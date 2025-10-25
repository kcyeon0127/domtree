"""Node schema definitions for DOMTree structures."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import copy

BBOX = Tuple[float, float, float, float]


@dataclasses.dataclass
class VisualCues:
    bbox: Optional[BBOX] = None
    font_size: Optional[float] = None
    font_weight: Optional[str] = None
    margin_top: Optional[float] = None
    margin_bottom: Optional[float] = None
    bg_color: Optional[str] = None
    column: Optional[int] = None

    def to_dict(self) -> Dict[str, float | str | None]:
        data: Dict[str, float | str | None] = {
            "bbox": list(self.bbox) if self.bbox else None,
            "font_size": self.font_size,
            "font_weight": self.font_weight,
            "margin_top": self.margin_top,
            "margin_bottom": self.margin_bottom,
            "bg_color": self.bg_color,
            "column": self.column,
        }
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, float | str | List[float] | None]) -> "VisualCues":
        bbox = data.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_tuple: Optional[BBOX] = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        else:
            bbox_tuple = None
        return cls(
            bbox=bbox_tuple,
            font_size=data.get("font_size"),
            font_weight=data.get("font_weight"),
            margin_top=data.get("margin_top"),
            margin_bottom=data.get("margin_bottom"),
            bg_color=data.get("bg_color"),
            column=data.get("column"),
        )


@dataclasses.dataclass
class NodeMetadata:
    node_type: str
    role: Optional[str] = None
    text_heading: Optional[str] = None
    heading_level: Optional[int] = None
    reading_order: Optional[int] = None
    dom_refs: List[str] = dataclasses.field(default_factory=list)
    visual_cues: VisualCues = dataclasses.field(default_factory=VisualCues)
    text_preview: Optional[str] = None
    language: Optional[str] = None
    notes: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "type": self.node_type,
            "role": self.role,
            "text_heading": self.text_heading,
            "heading_level": self.heading_level,
            "reading_order": self.reading_order,
            "dom_refs": self.dom_refs,
            "vis_cues": self.visual_cues.to_dict(),
            "text_preview": self.text_preview,
            "language": self.language,
            "notes": self.notes,
        }
        return {k: v for k, v in data.items() if v not in (None, [], {})}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "NodeMetadata":
        vis_cues = VisualCues.from_dict(data.get("vis_cues", {})) if isinstance(data.get("vis_cues"), dict) else VisualCues()
        return cls(
            node_type=str(data.get("type", "unknown")),
            role=data.get("role"),
            text_heading=data.get("text_heading"),
            heading_level=data.get("heading_level"),
            reading_order=data.get("reading_order"),
            dom_refs=[str(ref) for ref in data.get("dom_refs", [])] if isinstance(data.get("dom_refs"), list) else [],
            visual_cues=vis_cues,
            text_preview=data.get("text_preview"),
            language=data.get("language"),
            notes=data.get("notes", {}) if isinstance(data.get("notes"), dict) else {},
        )

    def copy(self) -> "NodeMetadata":
        return dataclasses.replace(
            self,
            dom_refs=list(self.dom_refs),
            visual_cues=VisualCues(**self.visual_cues.__dict__),
            notes=copy.deepcopy(self.notes),
        )


TREE_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://domtree.dev/schema/node.json",
    "title": "DomTree Node",
    "type": "object",
    "required": ["name", "metadata", "children"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "internal_thought_process": {"type": ["string", "null"]},
        "label": {"type": ["string", "null"]},
        "attributes": {"type": "object"},
        "metadata": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"type": "string", "minLength": 1},
                "role": {"type": ["string", "null"]},
                "text_heading": {"type": ["string", "null"]},
                "heading_level": {"type": ["integer", "null"]},
                "reading_order": {"type": ["integer", "null"]},
                "dom_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "vis_cues": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "font_size": {"type": "number"},
                        "font_weight": {"type": ["string", "null"]},
                        "margin_top": {"type": ["number", "null"]},
                        "margin_bottom": {"type": ["number", "null"]},
                        "bg_color": {"type": ["string", "null"]},
                        "column": {"type": ["integer", "null"]},
                    },
                    "additionalProperties": True,
                },
                "text_preview": {"type": ["string", "null"]},
                "language": {"type": ["string", "null"]},
                "notes": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "children": {
            "type": "array",
            "items": {"$ref": "#"},
        },
        "identifier": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}
