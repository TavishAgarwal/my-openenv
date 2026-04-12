---
title: InboxOps
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# InboxOps

## CI

[![Validate](https://github.com/TavishAgarwal/my-openenv/actions/workflows/validate.yml/badge.svg)](https://github.com/TavishAgarwal/my-openenv/actions/workflows/validate.yml)

**An OpenEnv-compliant AI agent training environment simulating an operations analyst's daily workload.**

InboxOps presents an AI agent with three interconnected tasks that mirror the routine work of a junior operations analyst: triaging an email inbox, routing support tickets to the right teams, and reconciling financial records for discrepancies. The environment is fully deterministic (seeded), self-contained, and designed to evaluate an agent's ability to classify, reason under time pressure, query structured data, and maintain context across related tasks.

Unlike single-task benchmarks, InboxOps tests **cross-task reasoning** — information discovered in one task (e.g., an invoice ID mentioned in a support ticket) is relevant to another (financial reconciliation). This makes it a strong signal for evaluating agentic memory and planning capabilities.

---

## Observation Space

| Field             | Type                    | Description                                               |
|-------------------|-------------------------|-----------------------------------------------------------|
| `inbox`           | `List[EmailMessage]`    | 25 emails with subject, body, sender, sender type, timestamp |
| `tickets`         | `List[SupportTicket]`   | 10 support tickets with description, customer tier, SLA times |
| `db_records`      | `List[PurchaseOrder]`   | 12 purchase orders (also queryable via SQL)                |
| `invoices`        | `List[Invoice]`         | 15 invoices with line items and PO references              |
| `current_task_id` | `str`                   | Current active task: `"task1"`, `"task2"`, or `"task3"`    |
| `step_count`      | `int`                   | Number of actions taken so far                             |
| `task_complete`   | `bool`                  | Whether the episode is finished                           |

---

## Action Space

| Action Type           | Key Fields                                          | When to Use                          |
|-----------------------|-----------------------------------------------------|--------------------------------------|
| `LabelEmailAction`   | `email_id`, `label`, `urgency`, `next_action`       | Task 1: classify and triage emails   |
| `RouteTicketAction`  | `ticket_id`, `team`, `escalate`, `draft_message`    | Task 2: route tickets to teams       |
| `QueryDatabaseAction`| `sql`                                                | Task 3: query invoices/POs via SQL   |
| `FlagDiscrepancyAction`| `invoice_id`, `po_id`, `discrepancy_type`, `explanation` | Task 3: flag a financial discrepancy |
| `SubmitReportAction` | `report` (dict)                                      | Task 3: submit final report (ends episode) |

---

## Tasks

### Task 1: Email Triage (Easy)

Classify 25 emails by category, urgency (1–5), and next action.

| Scoring Component    | Points  |
|----------------------|---------|
| Correct label        | +0.05   |
| Urgency exact match  | +0.03   |
| Urgency within ±1    | +0.02   |
| Correct next action  | +0.02   |
| Urgency off by ≥3    | −0.02   |
| **Max per email**    | **0.10**|

### Task 2: Support Ticket Routing (Medium)

Route 10 tickets to the correct team with escalation decisions.

| Scoring Component              | Points  |
|-------------------------------|---------|
| Correct team                  | +0.10   |
| Correct escalation            | +0.05   |
| Draft message with keywords   | +0.05   |
| SLA breach + no escalation    | −0.10   |
| Wrong team (enterprise)       | −0.05   |

### Task 3: Data Reconciliation (Hard)

Find 5 planted discrepancies across 15 invoices and 12 purchase orders.

| Scoring Component           | Points  |
|-----------------------------|---------|
| Correctly identified        | +0.15   |
| Correct discrepancy type    | +0.05   |
| False positive              | −0.10   |

Discrepancy types: `amount_mismatch`, `duplicate_line_item`, `missing_po`, `date_anomaly`

### Step-Waste Penalties

To discourage agents from burning steps without progress:

| Penalty                | Condition                                              | Effect                                                     |
|------------------------|--------------------------------------------------------|------------------------------------------------------------|
| Repeat-action penalty  | Consecutive identical action (same type + key fields)  | −0.01 added to step reward                                 |
| Step-decay factor      | After step 50 (configurable via `STEP_DECAY_THRESHOLD`)| Reward × `max(0.5, 1.0 − (step − 50) / 200 × 0.3)`       |
| Loop detection         | Same action taken ≥3 times total                       | Step returns −0.2 immediately (existing behaviour)         |

Both penalties are visible in `reward.breakdown` as `repeat_action_penalty` and `step_decay_factor`.

---

## Setup

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/inboxops.git
cd inboxops

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run baseline (requires HF_TOKEN)
export HF_TOKEN=hf_your_token_here
python baseline/run_baseline.py
```

### Docker

```bash
# Build
docker build -t inboxops .

# Run baseline
docker run -e HF_TOKEN=hf_your_token_here inboxops

# Run tests
docker run inboxops pytest tests/ -v
```

---

## Baseline Scores

| Task                 | GPT-4o | GPT-3.5 | Random |
|----------------------|--------|---------|--------|
| Email Triage         | ~0.81  | ~0.63   | ~0.21  |
| Ticket Routing       | ~0.74  | ~0.55   | ~0.14  |
| Data Reconciliation  | ~0.58  | ~0.31   | ~0.04  |

---

## Quick Start (Python)

```python
from environment.env import InboxOpsEnv
from environment.models import LabelEmailAction

env = InboxOpsEnv(seed=42)
obs = env.reset()

# Process the first email
email = obs.inbox[0]
action = LabelEmailAction(
    email_id=email.email_id,
    label="billing",
    urgency=3,
    next_action="reply",
)

obs, reward, done, info = env.step(action)
print(f"Reward: {reward.value}, Breakdown: {reward.breakdown}")
```

---

## Cross-Task Context

One invoice ID is shared between a support ticket (Task 2) and the financial records (Task 3). An agent that notices this cross-reference — e.g., "Issue with invoice INV-007" in a ticket description — and carries that context into reconciliation will naturally score higher.

---

## HF Spaces Deployment

To deploy on Hugging Face Spaces with the `openenv` tag:

1. Create a new Space (Docker SDK)
2. Upload the repository contents
3. Set the `HF_TOKEN` secret in Space settings
4. The Dockerfile will automatically run the baseline on launch
5. Tag the Space with `openenv` for discoverability

```yaml
# In your Space's README.md metadata:
tags:
  - openenv
  - enterprise
  - multi-task
```

---

## Architecture

```
inboxops/
├── openenv.yaml              # OpenEnv manifest
├── environment/
│   ├── env.py                # Core environment (reset/step/state)
│   ├── models.py             # Pydantic v2 data models
│   ├── generator.py          # Seeded synthetic data factory
│   └── graders/              # Per-task scoring logic
│       ├── email_grader.py
│       ├── ticket_grader.py
│       └── reconciliation_grader.py
├── tasks/                    # Task metadata & descriptions
├── baseline/                 # HF inference baseline agent
└── tests/                    # Comprehensive pytest suite
```

---

## License

MIT
