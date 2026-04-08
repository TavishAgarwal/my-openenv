"""Grader for Task 1 — Email Triage.

Scores a single LabelEmailAction against the ground-truth EmailMessage.
"""

from __future__ import annotations

from environment.models import EmailMessage, LabelEmailAction, StepReward


def grade_email_action(
    action: LabelEmailAction,
    ground_truth: EmailMessage,
) -> StepReward:
    """Grade an email labelling action.

    Scoring (max 0.10 per email):
        - label correct:             +0.05
        - urgency exact match:       +0.03
        - urgency within ±1:         +0.02  (if not exact)
        - next_action correct:       +0.02

    Penalty:
        - urgency off by ≥3:         −0.02
    """
    breakdown: dict[str, float] = {}
    total = 0.0

    # --- Label ---
    if action.label.lower().strip() == ground_truth.ground_truth_label.lower().strip():
        breakdown["label_correct"] = 0.05
        total += 0.05
    else:
        breakdown["label_correct"] = 0.0

    # --- Urgency ---
    diff = abs(action.urgency - ground_truth.ground_truth_urgency)
    if diff == 0:
        breakdown["urgency_exact"] = 0.03
        total += 0.03
    elif diff <= 1:
        breakdown["urgency_close"] = 0.02
        total += 0.02
    elif diff >= 3:
        breakdown["urgency_penalty"] = -0.02
        total -= 0.02

    # --- Next action ---
    if action.next_action.lower().strip() == ground_truth.ground_truth_next_action.lower().strip():
        breakdown["next_action_correct"] = 0.02
        total += 0.02
    else:
        breakdown["next_action_correct"] = 0.0

    clamped = max(-0.2, min(0.2, round(total, 4)))
    return StepReward(
        value=clamped,
        breakdown=breakdown,
        done=False,
        info={"email_id": action.email_id},
    )
