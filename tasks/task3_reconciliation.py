"""Task 3 — Financial Data Reconciliation.

Defines metadata and helpers for the reconciliation sub-task.
The agent can query an in-memory SQLite database, flag discrepancies,
and submit a final reconciliation report.
"""

from __future__ import annotations

TASK_ID = "task3"
TASK_NAME = "Data Reconciliation"
DIFFICULTY = "hard"
MAX_SCORE = 1.0

DISCREPANCY_TYPES = {
    "duplicate_line_item",
    "amount_mismatch",
    "missing_po",
    "date_anomaly",
}

DESCRIPTION = """
## Task 3: Data Reconciliation

You have 15 invoices and 12 purchase orders. Your job is to find and flag
discrepancies between them. There are exactly 5 planted discrepancies.

### Available Actions
1. **query_db**: Run SQL queries against the invoices and purchase_orders tables
2. **flag_discrepancy**: Flag a specific discrepancy with invoice_id, po_id, type, and explanation
3. **submit_report**: Submit your final reconciliation report (ends the episode)

### Discrepancy Types
- `amount_mismatch`: Invoice amount differs significantly from PO approved amount
- `duplicate_line_item`: Same line item appears more than once in an invoice
- `missing_po`: Invoice references a PO that doesn't exist
- `date_anomaly`: Invoice date is before PO approval date

### Scoring
- Each correctly identified discrepancy: +0.15
- Correct discrepancy type bonus: +0.05
- False positive: −0.10
- Score normalised to 0.0–1.0

### Database Schema
```sql
CREATE TABLE purchase_orders (
    po_id TEXT PRIMARY KEY,
    vendor_name TEXT,
    approved_amount REAL,
    approval_date TEXT,
    status TEXT
);

CREATE TABLE invoices (
    invoice_id TEXT PRIMARY KEY,
    vendor_name TEXT,
    amount REAL,
    date TEXT,
    po_number TEXT
);
```

### Tips
- Start with SQL queries to compare invoice amounts vs PO approved amounts
- Check for invoices whose po_number doesn't match any po_id
- Look for invoices dated before their PO's approval_date
- Cross-reference invoice IDs mentioned in support tickets from Task 2
"""


def get_task_info() -> dict:
    """Return a serialisable description of this task."""
    return {
        "task_id": TASK_ID,
        "name": TASK_NAME,
        "difficulty": DIFFICULTY,
        "max_score": MAX_SCORE,
        "discrepancy_types": sorted(DISCREPANCY_TYPES),
        "description": DESCRIPTION.strip(),
    }
