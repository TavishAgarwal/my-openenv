"""Task 1 — Email Triage.

Defines metadata and helpers for the email classification sub-task.
The agent must label each email by category, urgency, and next action.
"""

from __future__ import annotations

TASK_ID = "task1"
TASK_NAME = "Email Triage"
DIFFICULTY = "easy"
MAX_SCORE = 1.0

VALID_LABELS = {"billing", "onboarding", "outage", "spam", "general"}
VALID_NEXT_ACTIONS = {"reply", "escalate", "archive", "forward"}
URGENCY_RANGE = (1, 5)

DESCRIPTION = """
## Task 1: Email Triage

You have 25 emails in your inbox. For each email, you must:

1. **Classify** the email into one of: billing, onboarding, outage, spam, general
2. **Assign urgency** on a 1-5 scale (5 = most urgent)
3. **Determine next action**: reply, escalate, archive, or forward

### Scoring
- Correct label: +0.05
- Urgency exact match: +0.03 / within ±1: +0.02
- Correct next action: +0.02
- Urgency off by ≥3: −0.02

### Tips
- Outage emails are always high-urgency and should be escalated
- Spam should be archived with urgency 1
- Watch for VIP customer sender types — they may need priority handling
"""


def get_task_info() -> dict:
    """Return a serialisable description of this task."""
    return {
        "task_id": TASK_ID,
        "name": TASK_NAME,
        "difficulty": DIFFICULTY,
        "max_score": MAX_SCORE,
        "valid_labels": sorted(VALID_LABELS),
        "valid_next_actions": sorted(VALID_NEXT_ACTIONS),
        "urgency_range": URGENCY_RANGE,
        "description": DESCRIPTION.strip(),
    }
