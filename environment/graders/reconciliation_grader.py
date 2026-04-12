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

    Three-tier scoring per flag:
        - Correct (invoice_id, po_id) pair + correct type → full credit  (+0.20)
        - Correct (invoice_id, po_id) pair + wrong type   → partial credit (+0.05)
        - No pair match                                   → false positive  (−0.10)

    Final score = sum / (len(planted) × 0.20) → normalized 0.0–1.0
    """
    breakdown: dict[str, float] = {}
    raw_score = 0.0

    # Build lookup of planted discrepancies by (invoice_id, po_id) pair
    planted_by_pair: dict[tuple[str, str | None], PlantedDiscrepancy] = {}
    for p in planted:
        planted_by_pair[(p.invoice_id, p.po_id)] = p

    matched_pairs: set[tuple[str, str | None]] = set()

    for flag in submitted_flags:
        key = f"flag_{flag.invoice_id}"
        pair = (flag.invoice_id, flag.po_id)

        # Check if (invoice_id, po_id) matches a real planted discrepancy
        if pair in planted_by_pair and pair not in matched_pairs:
            gt = planted_by_pair[pair]
            matched_pairs.add(pair)

            if flag.discrepancy_type.lower().strip() == gt.discrepancy_type.value.lower().strip():
                # Full credit: pair match + correct type
                breakdown[f"{key}_identified"] = 0.15
                breakdown[f"{key}_correct_type"] = 0.05
                raw_score += 0.20
            else:
                # Partial credit: pair match, wrong type
                breakdown[f"{key}_identified_pair"] = 0.05
                breakdown[f"{key}_wrong_type"] = 0.0
                raw_score += 0.05
        else:
            # False positive: pair does not match any planted discrepancy
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
            "matched_count": len(matched_pairs),
        },
    )
