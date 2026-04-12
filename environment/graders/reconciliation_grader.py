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

    Compound discrepancies: some invoice/PO pairs may have multiple planted
    discrepancy types.  Both must be flagged for full credit on that pair.
    Partial credit (0.05) is awarded if only one of the required types is flagged.

    Final score = sum / (len(planted) × 0.20) → normalized 0.0–1.0
    """
    breakdown: dict[str, float] = {}
    raw_score = 0.0

    # Build lookup of planted discrepancies by (invoice_id, po_id) pair.
    # Some pairs may have multiple discrepancy types (compound discrepancies).
    planted_by_pair: dict[tuple[str, str | None], list[PlantedDiscrepancy]] = {}
    for p in planted:
        pair = (p.invoice_id, p.po_id)
        planted_by_pair.setdefault(pair, []).append(p)

    # Identify compound pairs (pairs with >1 planted discrepancy type)
    compound_pairs: dict[tuple[str, str | None], list[str]] = {}
    for pair, plist in planted_by_pair.items():
        types = [p.discrepancy_type.value for p in plist]
        if len(types) > 1:
            compound_pairs[pair] = types

    matched_pairs: set[tuple[str, str | None]] = set()
    # Track which compound pair types have been flagged
    compound_flagged: dict[tuple[str, str | None], list[str]] = {}

    for flag in submitted_flags:
        key = f"flag_{flag.invoice_id}"
        pair = (flag.invoice_id, flag.po_id)

        if pair in planted_by_pair and pair not in matched_pairs:
            plist = planted_by_pair[pair]
            gt_types = [p.discrepancy_type.value for p in plist]

            if pair in compound_pairs:
                # Compound discrepancy: accumulate flagged types
                compound_flagged.setdefault(pair, [])
                if flag.discrepancy_type.lower().strip() in gt_types:
                    compound_flagged[pair].append(flag.discrepancy_type.lower().strip())
                # Don't mark as matched yet; wait until all types checked
                # (scoring happens after the loop)
                continue
            else:
                # Single-type planted discrepancy — standard scoring
                gt = plist[0]
                matched_pairs.add(pair)

                if flag.discrepancy_type.lower().strip() == gt.discrepancy_type.value.lower().strip():
                    breakdown[f"{key}_identified"] = 0.15
                    breakdown[f"{key}_correct_type"] = 0.05
                    raw_score += 0.20
                else:
                    breakdown[f"{key}_identified_pair"] = 0.05
                    breakdown[f"{key}_wrong_type"] = 0.0
                    raw_score += 0.05
        else:
            # False positive: pair does not match any planted discrepancy
            # (or already matched for non-compound)
            if pair not in compound_pairs:
                breakdown[f"{key}_false_positive"] = -0.10
                raw_score -= 0.10

    # Score compound pairs after all flags have been processed
    for pair, required_types in compound_pairs.items():
        flagged_types = compound_flagged.get(pair, [])
        inv_id = pair[0]
        key = f"flag_{inv_id}_compound"

        if all(t in flagged_types for t in required_types):
            # Full credit: all required types were flagged
            breakdown[f"{key}_all_types"] = 0.20 * len(required_types)
            raw_score += 0.20 * len(required_types)
        elif any(t in flagged_types for t in required_types):
            # Partial credit: at least one type was flagged
            matched_count = sum(1 for t in required_types if t in flagged_types)
            breakdown[f"{key}_partial"] = 0.05 * matched_count
            raw_score += 0.05 * matched_count
        # else: no credit (unflagged compound pair)

        matched_pairs.add(pair)

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

