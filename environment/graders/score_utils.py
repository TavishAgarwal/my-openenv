"""Shared score normalisation utilities for InboxOps graders.

All final task scores must be strictly inside the open interval (0, 1).
Using 0.01 / 0.99 avoids floating-point edge cases that a tiny epsilon
like 1e-6 cannot reliably prevent after rounding.
"""

from __future__ import annotations

# Safe bounds
SCORE_FLOOR = 0.01
SCORE_CEIL = 0.99


def normalize_score(score: float | None) -> float:
    """Clamp *score* to the open interval (0, 1).

    Handles None, non-numeric, NaN, and ±inf inputs gracefully.
    Returns a value in [SCORE_FLOOR, SCORE_CEIL] so that no grader or
    inference path can ever emit exactly 0.0 or 1.0.
    """
    if score is None:
        return SCORE_FLOOR
    try:
        score = float(score)
    except (TypeError, ValueError):
        return SCORE_FLOOR
    # Guard against NaN / inf
    if score != score or score == float("inf") or score == float("-inf"):
        return SCORE_FLOOR
    return max(SCORE_FLOOR, min(SCORE_CEIL, score))
