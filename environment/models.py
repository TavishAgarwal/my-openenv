"""Pydantic v2 data models for the InboxOps environment.

Defines observation, action, and reward schemas used throughout the
environment, graders, and baseline agent.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SenderType(str, Enum):
    VIP_CUSTOMER = "vip_customer"
    INTERNAL_STAFF = "internal_staff"
    AUTOMATED_SYSTEM = "automated_system"
    UNKNOWN = "unknown"


class EmailMessage(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    sender_type: SenderType
    timestamp: datetime


class CustomerTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SupportTicket(BaseModel):
    ticket_id: str
    description: str
    customer_tier: CustomerTier
    created_at: datetime
    unresolved: bool
    sla_breach_at: datetime        # computed: created_at + tier_sla_hours


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    amount: float
    date: date
    po_number: Optional[str] = None
    line_items: List[dict]


class PurchaseOrder(BaseModel):
    po_id: str
    vendor_name: str
    approved_amount: float
    approval_date: date
    status: str                    # approved/pending/rejected


class DiscrepancyType(str, Enum):
    DUPLICATE_LINE_ITEM = "duplicate_line_item"
    AMOUNT_MISMATCH = "amount_mismatch"
    MISSING_PO = "missing_po"
    DATE_ANOMALY = "date_anomaly"
    VENDOR_MISMATCH = "vendor_mismatch"


class PlantedDiscrepancy(BaseModel):
    invoice_id: str
    po_id: Optional[str] = None
    discrepancy_type: DiscrepancyType
    description: str


# ---------------------------------------------------------------------------
# Internal ground-truth models (never exposed in Observation)
# ---------------------------------------------------------------------------

class EmailGroundTruth(BaseModel):
    """Answer key for a single email — used only by graders."""
    label: str                     # billing/onboarding/outage/spam/general
    urgency: int                   # 1-5
    next_action: str               # reply/escalate/archive/forward


class TicketGroundTruth(BaseModel):
    """Answer key for a single ticket — used only by graders."""
    team: str                      # billing/infra/product/account_management
    escalate: bool
    customer_tier: CustomerTier    # needed for enterprise wrong-team penalty
    sla_breach_at: datetime        # needed for SLA breach penalty


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    inbox: List[EmailMessage]
    tickets: List[SupportTicket]
    db_records: List[PurchaseOrder]
    invoices: List[Invoice]
    current_task_id: str           # "task1" | "task2" | "task3"
    step_count: int
    task_complete: bool


# ---------------------------------------------------------------------------
# Action models (discriminated union)
# ---------------------------------------------------------------------------

class LabelEmailAction(BaseModel):
    action_type: Literal["label_email"] = "label_email"
    email_id: str
    label: str
    urgency: int                   # 1-5
    next_action: str               # reply/escalate/archive/forward


class RouteTicketAction(BaseModel):
    action_type: Literal["route_ticket"] = "route_ticket"
    ticket_id: str
    team: str
    escalate: bool
    draft_message: Optional[str] = None


class QueryDatabaseAction(BaseModel):
    action_type: Literal["query_db"] = "query_db"
    sql: str                       # executed against SQLite in-memory DB


class FlagDiscrepancyAction(BaseModel):
    action_type: Literal["flag_discrepancy"] = "flag_discrepancy"
    invoice_id: str
    po_id: Optional[str] = None
    discrepancy_type: str
    explanation: str


class SubmitReportAction(BaseModel):
    action_type: Literal["submit_report"] = "submit_report"
    report: dict                   # must match ReconciliationReport schema


Action = Union[
    LabelEmailAction,
    RouteTicketAction,
    QueryDatabaseAction,
    FlagDiscrepancyAction,
    SubmitReportAction,
]


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class StepReward(BaseModel):
    value: float                   # incremental reward for this step
    breakdown: Dict[str, float]    # {"email_correct": 0.05, "penalty": 0.0, …}
    done: bool
    info: dict
