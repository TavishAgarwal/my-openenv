"""Grader for Task 2 — Support Ticket Routing.

Scores a single RouteTicketAction against the ground-truth SupportTicket.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from environment.models import CustomerTier, RouteTicketAction, StepReward, SupportTicket

# Keywords that a good draft message should contain per team
_TEAM_KEYWORDS: dict[str, list[str]] = {
    "billing": ["invoice", "charge", "payment", "refund", "billing", "account"],
    "infra": ["latency", "outage", "server", "deploy", "pipeline", "database", "api"],
    "product": ["feature", "request", "bug", "integration", "update", "export"],
    "account_management": ["upgrade", "account", "tier", "enterprise", "sla", "gdpr", "deletion"],
}


def grade_ticket_action(
    action: RouteTicketAction,
    ground_truth: SupportTicket,
    current_time: datetime,
) -> StepReward:
    """Grade a ticket routing action.

    Scoring:
        - team correct:              +0.10
        - escalate correct:          +0.05
        - draft_message keywords:    +0.05  (if ≥1 expected keyword present)

    Penalty:
        - SLA breach within 30 min AND escalate=False:  −0.10
        - Wrong team for enterprise customer:            −0.05
    """
    breakdown: dict[str, float] = {}
    total = 0.0

    # --- Team ---
    team_correct = action.team.lower().strip() == ground_truth.ground_truth_team.lower().strip()
    if team_correct:
        breakdown["team_correct"] = 0.10
        total += 0.10
    else:
        breakdown["team_correct"] = 0.0
        # Extra penalty for enterprise
        if ground_truth.customer_tier == CustomerTier.ENTERPRISE:
            breakdown["enterprise_wrong_team_penalty"] = -0.05
            total -= 0.05

    # --- Escalate ---
    if action.escalate == ground_truth.ground_truth_escalate:
        breakdown["escalate_correct"] = 0.05
        total += 0.05
    else:
        breakdown["escalate_correct"] = 0.0

    # SLA breach penalty
    time_to_breach = (ground_truth.sla_breach_at - current_time).total_seconds()
    if time_to_breach <= 1800 and not action.escalate:  # 30 minutes
        breakdown["sla_breach_penalty"] = -0.10
        total -= 0.10

    # --- Draft message keyword check ---
    if action.draft_message:
        expected_keywords = _TEAM_KEYWORDS.get(
            ground_truth.ground_truth_team.lower(), []
        )
        msg_lower = action.draft_message.lower()
        found = any(kw in msg_lower for kw in expected_keywords)
        if found:
            breakdown["draft_keywords"] = 0.05
            total += 0.05
        else:
            breakdown["draft_keywords"] = 0.0
    else:
        breakdown["draft_keywords"] = 0.0

    clamped = max(-0.2, min(0.2, round(total, 4)))
    return StepReward(
        value=clamped,
        breakdown=breakdown,
        done=False,
        info={"ticket_id": action.ticket_id},
    )
