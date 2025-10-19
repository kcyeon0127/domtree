"""Reading order alignment using a Needleman–Wunsch dynamic program."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from ..tree import TreeNode


@dataclass
class AlignmentResult:
    score: float
    normalized_score: float
    alignment: List[Tuple[Optional[str], Optional[str]]]
    reference_sequence: List[str]
    candidate_sequence: List[str]


_DEF_THRESHOLD = 0.55
_MATCH_REWARD = 1.0
_MISMATCH_PENALTY = -0.65
_GAP_PENALTY = -0.6


def _reading_sequence(node: TreeNode) -> List[str]:
    sequence: List[str] = []

    def _dfs(current: TreeNode) -> None:
        label = current.metadata.text_preview or current.label or current.name
        if label:
            sequence.append(str(label))
        for child in current.children:
            _dfs(child)

    _dfs(node)
    return sequence


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def reading_order_alignment(reference: TreeNode, candidate: TreeNode) -> AlignmentResult:
    ref_seq = _reading_sequence(reference)
    cand_seq = _reading_sequence(candidate)
    n = len(ref_seq)
    m = len(cand_seq)
    if n == 0 or m == 0:
        return AlignmentResult(score=0.0, normalized_score=0.0, alignment=[], reference_sequence=ref_seq, candidate_sequence=cand_seq)

    score_matrix = [[0.0] * (m + 1) for _ in range(n + 1)]
    pointer = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score_matrix[i][0] = i * _GAP_PENALTY
        pointer[i][0] = "up"
    for j in range(1, m + 1):
        score_matrix[0][j] = j * _GAP_PENALTY
        pointer[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = _similarity(ref_seq[i - 1], cand_seq[j - 1])
            if sim >= _DEF_THRESHOLD:
                diag_score = score_matrix[i - 1][j - 1] + _MATCH_REWARD * sim
            else:
                diag_score = score_matrix[i - 1][j - 1] + _MISMATCH_PENALTY * (1 - sim)
            up_score = score_matrix[i - 1][j] + _GAP_PENALTY
            left_score = score_matrix[i][j - 1] + _GAP_PENALTY
            best_score = max(diag_score, up_score, left_score)
            score_matrix[i][j] = best_score
            if best_score == diag_score:
                pointer[i][j] = "diag"
            elif best_score == up_score:
                pointer[i][j] = "up"
            else:
                pointer[i][j] = "left"

    # Traceback
    alignment: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = pointer[i][j]
        if move == "diag":
            alignment.append((ref_seq[i - 1], cand_seq[j - 1]))
            i -= 1
            j -= 1
        elif move == "up":
            alignment.append((ref_seq[i - 1], None))
            i -= 1
        else:
            alignment.append((None, cand_seq[j - 1]))
            j -= 1
    alignment.reverse()

    score = score_matrix[n][m]
    normalized = score / (max(n, m) * _MATCH_REWARD)
    return AlignmentResult(
        score=score,
        normalized_score=normalized,
        alignment=alignment,
        reference_sequence=ref_seq,
        candidate_sequence=cand_seq,
    )
