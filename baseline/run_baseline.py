"""Rule-based baseline agent for InboxOps.

Runs a simple keyword-matching heuristic agent through all 3 tasks.
No LLM required — purely deterministic.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import InboxOpsEnv
from environment.graders import normalize_score, SCORE_FLOOR
from environment.models import (
    FlagDiscrepancyAction,
    LabelEmailAction,
    QueryDatabaseAction,
    RouteTicketAction,
    SubmitReportAction,
)


# ---------------------------------------------------------------------------
# Task 1: Email triage via keyword matching
# ---------------------------------------------------------------------------

def _classify_email(subject: str, body: str) -> tuple[str, int, str]:
    """Return (label, urgency, next_action) using keyword matching."""
    text = (subject + " " + body).lower()

    if any(kw in text for kw in ("invoice", "charge", "payment", "refund", "billing")):
        return "billing", 3, "reply"
    if any(kw in text for kw in ("outage", "503", "error", "alert", "connection pool")):
        return "outage", 3, "escalate"
    if any(kw in text for kw in ("new hire", "onboarding", "access", "welcome", "new team")):
        return "onboarding", 3, "forward"
    if any(kw in text for kw in ("click", "prize", "urgent verify", "won a", "gift card",
                                   "$$", "wire transfer", "guaranteed returns",
                                   "track it here", "update your account")):
        return "spam", 3, "archive"
    return "general", 3, "reply"


# ---------------------------------------------------------------------------
# Task 2: Ticket routing via keyword matching
# ---------------------------------------------------------------------------

def _classify_ticket(description: str, customer_tier: str) -> tuple[str, bool]:
    """Return (team, escalate) using keyword matching."""
    text = description.lower()

    if any(kw in text for kw in ("charge", "invoice", "billing", "charged")):
        team = "billing"
    elif any(kw in text for kw in ("latency", "pipeline", "503", "webhook", "api update")):
        team = "infra"
    elif any(kw in text for kw in ("feature", "export", "integration")):
        team = "product"
    else:
        team = "account_management"

    escalate = customer_tier == "enterprise"
    return team, escalate


# ---------------------------------------------------------------------------
# Task 3: Reconciliation via simple amount comparison
# ---------------------------------------------------------------------------

def _run_reconciliation(env: InboxOpsEnv) -> None:
    """Query DB, flag mismatches, submit report."""
    # Query all invoices and POs
    query = QueryDatabaseAction(
        sql=(
            "SELECT i.invoice_id, i.vendor_name AS inv_vendor, i.amount, i.date, i.po_number, "
            "p.po_id, p.vendor_name AS po_vendor, p.approved_amount, p.approval_date "
            "FROM invoices i "
            "LEFT JOIN purchase_orders p ON i.po_number = p.po_id"
        )
    )
    _, _, _, info = env.step(query)
    rows = info.get("query_result", {}).get("rows", [])

    flagged = []
    for row in rows:
        inv_id, inv_vendor, amount, inv_date, po_number, po_id, po_vendor, approved_amt, approval_date = row

        if po_number and po_id is None:
            # Referenced PO doesn't exist
            flag = FlagDiscrepancyAction(
                invoice_id=inv_id,
                po_id=po_number,
                discrepancy_type="missing_po",
                explanation=f"Invoice {inv_id} references PO {po_number} which does not exist.",
            )
            env.step(flag)
            flagged.append(inv_id)
            continue

        if approved_amt is not None and amount != approved_amt:
            # Check if difference is > 5% (to avoid red herrings)
            pct_diff = abs(amount - approved_amt) / approved_amt if approved_amt else 0
            if pct_diff > 0.05:
                flag = FlagDiscrepancyAction(
                    invoice_id=inv_id,
                    po_id=po_id,
                    discrepancy_type="amount_mismatch",
                    explanation=f"Invoice amount {amount} != PO approved {approved_amt} (diff {pct_diff*100:.1f}%).",
                )
                env.step(flag)
                flagged.append(inv_id)

    # Submit report
    report = SubmitReportAction(
        report={
            "discrepancies": [{"invoice_id": fid} for fid in flagged],
            "summary": f"Found {len(flagged)} discrepancies via rule-based matching.",
        }
    )
    env.step(report)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_tasks(seed: int = 42) -> dict:
    """Run all 3 tasks with a rule-based agent. Returns results dict."""
    env = InboxOpsEnv(seed=seed)
    obs = env.reset()

    total_steps = 0

    # --- Task 1: Email triage ---
    for email in obs.inbox:
        label, urgency, next_action = _classify_email(email.subject, email.body)
        action = LabelEmailAction(
            email_id=email.email_id,
            label=label,
            urgency=urgency,
            next_action=next_action,
        )
        obs, reward, done, info = env.step(action)
        total_steps += 1

    task1_score = env._scores.get("task1", SCORE_FLOOR)

    # --- Task 2: Ticket routing ---
    for ticket in obs.tickets:
        team, escalate = _classify_ticket(ticket.description, ticket.customer_tier.value)
        action = RouteTicketAction(
            ticket_id=ticket.ticket_id,
            team=team,
            escalate=escalate,
            draft_message="Routing to appropriate team for review.",
        )
        obs, reward, done, info = env.step(action)
        total_steps += 1

    task2_score = env._scores.get("task2", SCORE_FLOOR)

    # --- Task 3: Reconciliation ---
    _run_reconciliation(env)
    total_steps += 2  # at least query + submit

    task3_score = env._scores.get("task3", SCORE_FLOOR)

    results = {
        "task1": {"score": round(normalize_score(task1_score), 4), "steps": len(obs.inbox)},
        "task2": {"score": round(normalize_score(task2_score), 4), "steps": len(obs.tickets)},
        "task3": {"score": round(normalize_score(task3_score), 4), "steps": total_steps},
    }

    return results


def main():
    results = run_all_tasks(seed=42)

    # Print summary table
    print("\n" + "=" * 50)
    print("  InboxOps Rule-Based Baseline Results")
    print("=" * 50)
    print(f"  {'Task':<25} {'Score':>8} {'Steps':>8}")
    print("-" * 50)
    for task_id in ("task1", "task2", "task3"):
        r = results[task_id]
        print(f"  {task_id:<25} {r['score']:>8.4f} {r['steps']:>8}")
    print("=" * 50)

    # Save results
    out_path = Path(__file__).parent / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
