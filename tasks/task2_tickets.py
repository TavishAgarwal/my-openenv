"""Task 2 — Support Ticket Routing.

Defines metadata and helpers for the ticket routing sub-task.
The agent must assign each ticket to the correct team, decide whether
to escalate, and optionally draft a message.
"""

from __future__ import annotations

TASK_ID = "task2"
TASK_NAME = "Support Ticket Routing"
DIFFICULTY = "medium"
MAX_SCORE = 1.0

VALID_TEAMS = {"billing", "infra", "product", "account_management"}

DESCRIPTION = """
## Task 2: Support Ticket Routing

You have 10 support tickets to process. For each ticket, you must:

1. **Route** it to the correct team: billing, infra, product, or account_management
2. **Decide escalation**: set escalate=true if the situation warrants it
3. **Draft a message** (optional but scored): include relevant keywords

### Scoring
- Correct team: +0.10
- Correct escalation decision: +0.05
- Draft message with relevant keywords: +0.05

### Penalties
- SLA breach within 30 min AND escalate=False: −0.10
- Wrong team for enterprise customer: −0.05

### Tips
- Enterprise customers have a 4-hour SLA — check breach times carefully
- Tickets mentioning invoice IDs may connect to Task 3 reconciliation data
- Always escalate when SLA breach is imminent
"""


def get_task_info() -> dict:
    """Return a serialisable description of this task."""
    return {
        "task_id": TASK_ID,
        "name": TASK_NAME,
        "difficulty": DIFFICULTY,
        "max_score": MAX_SCORE,
        "valid_teams": sorted(VALID_TEAMS),
        "description": DESCRIPTION.strip(),
    }
