"""Grader for Task 3 — Data Reconciliation.

Provides two functions:
  - grade_query_action: execute arbitrary SQL against the in-memory SQLite DB
  - grade_report_submission: score flagged discrepancies against planted ones
"""

from __future__ import annotations

import sqlite3
from typing import List

from environment.models import (
    FlagDiscrepancyAction,
    PlantedDiscrepancy,
    StepReward,
)


def grade_query_action(sql: str, conn: sqlite3.Connection) -> dict:
    """Execute a SQL query on the in-memory database.

    Returns:
        {"rows": List[tuple], "columns": List[str], "error": None | str}
    """
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return {"rows": [list(r) for r in rows], "columns": columns, "error": None}
    except Exception as exc:
        return {"rows": [], "columns": [], "error": str(exc)}


def grade_report_submission(
    submitted_flags: List[FlagDiscrepancyAction],
    planted: List[PlantedDiscrepancy],
) -> StepReward:
    """Grade a set of flagged discrepancies against planted ground truth.

    Deterministic scoring:
        - Each planted discrepancy correctly identified (by invoice_id match): +0.15
        - Each correctly typed (discrepancy_type also matches):               +0.05 bonus
        - False positive (flagged but not planted):                           −0.10
        - Final score = sum / (len(planted) * 0.20) → normalized 0.0–1.0
    """
    breakdown: dict[str, float] = {}
    raw_score = 0.0

    # Build lookup of planted discrepancies by invoice_id
    planted_by_invoice: dict[str, PlantedDiscrepancy] = {}
    for p in planted:
        planted_by_invoice[p.invoice_id] = p

    matched_invoice_ids: set[str] = set()

    for flag in submitted_flags:
        key = f"flag_{flag.invoice_id}"
        if flag.invoice_id in planted_by_invoice and flag.invoice_id not in matched_invoice_ids:
            gt = planted_by_invoice[flag.invoice_id]
            matched_invoice_ids.add(flag.invoice_id)

            # Correctly identified
            breakdown[f"{key}_identified"] = 0.15
            raw_score += 0.15

            # Type bonus
            if flag.discrepancy_type.lower().strip() == gt.discrepancy_type.value.lower().strip():
                breakdown[f"{key}_type_bonus"] = 0.05
                raw_score += 0.05
            else:
                breakdown[f"{key}_type_bonus"] = 0.0
        else:
            # False positive
            breakdown[f"{key}_false_positive"] = -0.10
            raw_score -= 0.10

    # Normalize: max possible = len(planted) × 0.20
    max_possible = len(planted) * 0.20 if planted else 1.0
    # Clamp final score to [0.0, 1.0]
    normalized = max(0.0, min(1.0, raw_score / max_possible))

    breakdown["raw_score"] = round(raw_score, 4)
    breakdown["normalized_score"] = round(normalized, 4)

    return StepReward(
        value=round(normalized, 4),
        breakdown=breakdown,
        done=True,
        info={
            "planted_count": len(planted),
            "flagged_count": len(submitted_flags),
            "matched_count": len(matched_invoice_ids),
        },
    )
