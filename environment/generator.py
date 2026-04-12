"""Seeded synthetic data factory for InboxOps episodes.

Uses Faker with a fixed seed to guarantee full reproducibility across
identical seed values.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Dict, List

from faker import Faker

from environment.models import (
    CustomerTier,
    DiscrepancyType,
    EmailGroundTruth,
    EmailMessage,
    Invoice,
    PlantedDiscrepancy,
    PurchaseOrder,
    SenderType,
    SupportTicket,
    TicketGroundTruth,
)


# ---------------------------------------------------------------------------
# Email templates by category
# ---------------------------------------------------------------------------

_EMAIL_TEMPLATES: Dict[str, List[dict]] = {
    "billing": [
        {"subj": "Invoice discrepancy on account #{acct}", "body": "Hi, I noticed a charge of ${amt} on my latest invoice that doesn't match the agreed rate. Could you look into this? My account number is {acct}. Thanks, {name}"},
        {"subj": "Payment failed – retry needed", "body": "Our automated payment for subscription {acct} failed this morning. Please update the payment method on file or retry the charge. Regards, {name}"},
        {"subj": "Refund request for duplicate charge", "body": "I was charged twice for the same service on {date}. Please process a refund for the duplicate amount of ${amt}. Account: {acct}. – {name}"},
        {"subj": "Billing cycle change request", "body": "We'd like to switch from monthly to annual billing for account {acct}. Can you confirm the prorated amount? Best, {name}"},
    ],
    "onboarding": [
        {"subj": "New team member setup – {name}", "body": "Please provision access for our new hire {name} starting {date}. They'll need access to the dashboard and reporting tools. Team: Engineering."},
        {"subj": "Onboarding checklist incomplete", "body": "Hi, I'm {name} and I just joined. My onboarding checklist shows 3 pending items: SSO setup, Slack channels, and VPN access. Can someone help?"},
        {"subj": "Welcome pack not received", "body": "I started last week but haven't received the welcome pack or my hardware. My employee ID is {acct}. Please advise. – {name}"},
        {"subj": "Training environment access", "body": "Hi team, {name} here. I need access to the staging/training environment for my onboarding exercises. Could you create an account for me?"},
        {"subj": "Onboarding feedback", "body": "Just completed onboarding week 1. Here's my feedback: the documentation for API setup is outdated (refers to v2 but we're on v4). Also the Slack bot invite link is broken. – {name}"},
    ],
    "outage": [
        {"subj": "[URGENT] Production API returning 503s", "body": "Our monitoring detected that the /api/v3/data endpoint has been returning 503 errors for the past 15 minutes. Affected region: us-east-1. Customers are reporting data sync failures. Immediate action required."},
        {"subj": "Database connection pool exhausted", "body": "Alert: The primary database connection pool has hit its limit of 500 connections. Read replicas are also showing high latency (>2s). This is impacting all services downstream."},
        {"subj": "CDN cache invalidation stuck", "body": "The CDN cache purge job that was triggered at {date} is stuck in a pending state. Static assets are serving stale content. Multiple customers have reported seeing outdated UI. – SRE Team"},
    ],
    "spam": [
        {"subj": "Congratulations! You've won a $1000 gift card!", "body": "Click here to claim your prize immediately! Limited time offer. No purchase necessary. Act now before it expires!!!!"},
        {"subj": "URGENT: Update your account information NOW", "body": "Dear valued customer, your account will be suspended unless you verify your information within 24 hours. Click the link below to confirm your identity."},
        {"subj": "Make $$$ working from home!!!", "body": "Earn $5000/week with this simple trick! No experience needed! Join thousands of satisfied workers. Reply to this email for details."},
        {"subj": "Your package is waiting – track it here", "body": "You have a package waiting for delivery. Click the tracking link below to schedule delivery. If you don't respond within 48 hours, the package will be returned."},
        {"subj": "Exclusive investment opportunity – 500% returns!", "body": "Dear investor, we have an exclusive pre-IPO opportunity with guaranteed returns of 500%. Minimum investment just $100. Wire transfer details enclosed."},
    ],
    "general": [
        {"subj": "Q3 planning meeting – agenda items needed", "body": "Hi team, please submit your agenda items for the Q3 planning meeting by EOD Friday. We'll be covering roadmap priorities, hiring plans, and budget allocation. – {name}"},
        {"subj": "Office supplies order", "body": "Placing the monthly office supplies order this Wednesday. Reply with anything you need: monitors, keyboards, headsets, etc. Budget remaining: $2,400. – {name}"},
        {"subj": "Parking garage maintenance notice", "body": "The parking garage on Level B2 will be closed for maintenance from {date} to the following Monday. Please use the surface lot during this period. – Facilities"},
        {"subj": "Team lunch on Friday", "body": "Hi everyone, we're organizing a team lunch this Friday at noon. Please fill out the dietary preferences form by Thursday. Location TBD. – {name}"},
        {"subj": "Updated PTO policy", "body": "Please review the updated PTO policy effective next quarter. Key changes: rollover limit increased to 10 days, new sick leave category added. Full document attached. – HR"},
        {"subj": "Vendor contract renewal reminder", "body": "Reminder: The contract with {name} Services expires on {date}. Please review the renewal terms and confirm whether we should proceed or explore alternatives. – Procurement"},
        {"subj": "Monthly newsletter – Operations Update", "body": "Here's your monthly ops update: uptime was 99.95%, 3 incidents resolved, 2 new automations deployed. Full report in the wiki. – {name}"},
        {"subj": "Conference room booking conflict", "body": "There's a double-booking for Conference Room A on Thursday 2-4pm. Can one of the teams switch to Room C? Please coordinate. – {name}"},
    ],
}

_NEXT_ACTIONS = {
    "billing": "reply",
    "onboarding": "forward",
    "outage": "escalate",
    "spam": "archive",
    "general": "reply",
}

# Teams for ticket routing
_TICKET_TEAMS = ["billing", "infra", "product", "account_management"]

_TICKET_TEMPLATES = [
    "Customer reports being charged incorrectly for {product}. Invoice {inv_ref} may be related.",
    "Cannot access dashboard after password reset. Customer tier: {tier}.",
    "Latency spikes on the data pipeline affecting {product} service.",
    "Feature request: customer wants bulk export for {product}.",
    "Account {acct} flagged for unusual login activity from multiple IPs.",
    "Customer requesting upgrade from {tier} to Enterprise. Needs pricing.",
    "Webhook delivery failures to customer endpoint for {product}.",
    "SLA breach warning for account {acct} – response time exceeded.",
    "Integration with {product} broken after latest API update.",
    "Customer requesting data deletion under GDPR for account {acct}.",
]


def generate_episode(seed: int = 42) -> dict:
    """Generate a complete episode's worth of synthetic data.

    Returns a dict with keys:
        emails, tickets, invoices, purchase_orders,
        planted_discrepancies, shared_invoice_id
    """
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)

    now = datetime(2025, 3, 15, 9, 0, 0)  # fixed "now" for determinism

    # ------------------------------------------------------------------
    # 1. Emails (25 total)
    # ------------------------------------------------------------------
    label_counts = {
        "billing": 4,
        "onboarding": 5,
        "outage": 3,
        "spam": 5,
        "general": 8,
    }

    # Urgency distribution: 3× urgency=5, 5× urgency=4, rest ≤3
    urgency_pool: List[int] = [5, 5, 5, 4, 4, 4, 4, 4] + [rng.randint(1, 3) for _ in range(17)]
    rng.shuffle(urgency_pool)

    sender_types = [SenderType.VIP_CUSTOMER, SenderType.INTERNAL_STAFF,
                    SenderType.AUTOMATED_SYSTEM, SenderType.UNKNOWN]

    emails: List[EmailMessage] = []
    email_ground_truths: Dict[str, EmailGroundTruth] = {}
    urgency_idx = 0
    for label, count in label_counts.items():
        templates = _EMAIL_TEMPLATES[label]
        for i in range(count):
            tmpl = templates[i % len(templates)]
            fmt_args = {
                "name": fake.name(),
                "acct": f"ACCT-{fake.random_number(digits=6, fix_len=True)}",
                "amt": f"{rng.uniform(50, 5000):.2f}",
                "date": fake.date_this_year().isoformat(),
            }
            subject = tmpl["subj"].format(**fmt_args)
            body = tmpl["body"].format(**fmt_args)

            # Outage emails get specific sender type
            if label == "outage":
                st = SenderType.AUTOMATED_SYSTEM
            elif label == "spam":
                st = SenderType.UNKNOWN
            else:
                st = rng.choice(sender_types)

            urg = urgency_pool[urgency_idx]
            # Override: outage always ≥4, spam always 1
            if label == "outage":
                urg = max(urg, 4)
            elif label == "spam":
                urg = 1
            urgency_idx += 1

            email_id = f"EMAIL-{len(emails)+1:03d}"

            emails.append(EmailMessage(
                email_id=email_id,
                subject=subject,
                body=body,
                sender=fake.email(),
                sender_type=st,
                timestamp=now - timedelta(minutes=rng.randint(5, 720)),
            ))

            email_ground_truths[email_id] = EmailGroundTruth(
                label=label,
                urgency=urg,
                next_action=_NEXT_ACTIONS[label],
            )

    rng.shuffle(emails)

    # ------------------------------------------------------------------
    # 2. Purchase orders (12 POs)
    # ------------------------------------------------------------------
    vendor_names = [fake.company() for _ in range(8)]  # 8 unique vendors
    purchase_orders: List[PurchaseOrder] = []
    for i in range(12):
        vendor = vendor_names[i % len(vendor_names)]
        approved_amt = round(rng.uniform(1000, 50000), 2)
        approval_dt = fake.date_between(start_date="-90d", end_date="-10d")
        purchase_orders.append(PurchaseOrder(
            po_id=f"PO-{i+1:03d}",
            vendor_name=vendor,
            approved_amount=approved_amt,
            approval_date=approval_dt,
            status=rng.choice(["approved", "approved", "approved", "pending"]),
        ))

    # ------------------------------------------------------------------
    # 3. Invoices (15 invoices) — some mapped to POs, some not
    # ------------------------------------------------------------------
    invoices: List[Invoice] = []
    for i in range(15):
        if i < 12:
            # Map to a PO
            po = purchase_orders[i]
            vendor = po.vendor_name
            po_number = po.po_id
            base_amount = po.approved_amount
            inv_date = po.approval_date + timedelta(days=rng.randint(5, 30))
        else:
            vendor = rng.choice(vendor_names)
            po_number = None
            base_amount = round(rng.uniform(500, 20000), 2)
            inv_date = fake.date_between(start_date="-60d", end_date="today")

        n_items = rng.randint(1, 4)
        line_items = []
        remaining = base_amount
        for li in range(n_items):
            if li == n_items - 1:
                item_amt = round(remaining, 2)
            else:
                item_amt = round(rng.uniform(100, remaining / 2), 2)
                remaining -= item_amt
            line_items.append({
                "description": fake.bs(),
                "quantity": rng.randint(1, 10),
                "unit_price": round(item_amt / rng.randint(1, 10), 2),
                "total": item_amt,
            })

        invoices.append(Invoice(
            invoice_id=f"INV-{i+1:03d}",
            vendor_name=vendor,
            amount=base_amount,
            date=inv_date,
            po_number=po_number,
            line_items=line_items,
        ))

    # ------------------------------------------------------------------
    # 4. Planted discrepancies (exactly 8)
    # ------------------------------------------------------------------
    planted: List[PlantedDiscrepancy] = []

    # 2× AMOUNT_MISMATCH — invoice amount differs from PO by 10-25%
    mismatch_indices = rng.sample(range(12), 2)  # pick 2 invoices that have POs
    for idx in mismatch_indices:
        inv = invoices[idx]
        po = purchase_orders[idx]
        factor = 1 + rng.uniform(0.10, 0.25) * rng.choice([1, -1])
        new_amount = round(po.approved_amount * factor, 2)
        # Mutate invoice amount
        invoices[idx] = inv.model_copy(update={"amount": new_amount})
        planted.append(PlantedDiscrepancy(
            invoice_id=inv.invoice_id,
            po_id=po.po_id,
            discrepancy_type=DiscrepancyType.AMOUNT_MISMATCH,
            description=(
                f"Invoice {inv.invoice_id} amount ${new_amount:.2f} differs from "
                f"PO {po.po_id} approved amount ${po.approved_amount:.2f} "
                f"(off by {abs(factor - 1) * 100:.1f}%)"
            ),
        ))

    # 1× DUPLICATE_LINE_ITEM — same line item appears twice
    dup_idx = rng.choice([i for i in range(15) if i not in mismatch_indices])
    inv = invoices[dup_idx]
    if inv.line_items:
        dup_item = inv.line_items[0].copy()
        new_items = inv.line_items + [dup_item]
        new_amount = round(inv.amount + dup_item["total"], 2)
        invoices[dup_idx] = inv.model_copy(update={
            "line_items": new_items,
            "amount": new_amount,
        })
        planted.append(PlantedDiscrepancy(
            invoice_id=inv.invoice_id,
            po_id=inv.po_number,
            discrepancy_type=DiscrepancyType.DUPLICATE_LINE_ITEM,
            description=(
                f"Invoice {inv.invoice_id} contains a duplicate line item: "
                f"'{dup_item['description']}' appears twice"
            ),
        ))

    # 1× MISSING_PO — invoice references a PO that doesn't exist
    missing_idx = rng.choice([i for i in range(12, 15)])
    inv = invoices[missing_idx]
    fake_po = f"PO-{rng.randint(900, 999)}"
    invoices[missing_idx] = inv.model_copy(update={"po_number": fake_po})
    planted.append(PlantedDiscrepancy(
        invoice_id=inv.invoice_id,
        po_id=fake_po,
        discrepancy_type=DiscrepancyType.MISSING_PO,
        description=(
            f"Invoice {inv.invoice_id} references PO {fake_po} which does "
            f"not exist in the purchase orders database"
        ),
    ))

    # 1× DATE_ANOMALY — invoice date is before PO approval date
    date_idx = rng.choice([
        i for i in range(12)
        if i not in mismatch_indices and i != dup_idx
    ])
    inv = invoices[date_idx]
    po = purchase_orders[date_idx]
    bad_date = po.approval_date - timedelta(days=rng.randint(10, 60))
    invoices[date_idx] = inv.model_copy(update={"date": bad_date})
    planted.append(PlantedDiscrepancy(
        invoice_id=inv.invoice_id,
        po_id=po.po_id,
        discrepancy_type=DiscrepancyType.DATE_ANOMALY,
        description=(
            f"Invoice {inv.invoice_id} dated {bad_date.isoformat()} is before "
            f"PO {po.po_id} approval date {po.approval_date.isoformat()}"
        ),
    ))

    # Track all used indices so far
    used_indices = set(mismatch_indices) | {dup_idx, date_idx}

    # 1× VENDOR_MISMATCH — invoice vendor differs from PO vendor
    vendor_mismatch_idx = rng.choice([
        i for i in range(12)
        if i not in used_indices
    ])
    inv = invoices[vendor_mismatch_idx]
    po = purchase_orders[vendor_mismatch_idx]
    wrong_vendor = rng.choice([v for v in vendor_names if v != po.vendor_name])
    invoices[vendor_mismatch_idx] = inv.model_copy(update={"vendor_name": wrong_vendor})
    planted.append(PlantedDiscrepancy(
        invoice_id=inv.invoice_id,
        po_id=po.po_id,
        discrepancy_type=DiscrepancyType.VENDOR_MISMATCH,
        description=(
            f"Invoice {inv.invoice_id} vendor '{wrong_vendor}' differs from "
            f"PO {po.po_id} vendor '{po.vendor_name}'"
        ),
    ))
    used_indices.add(vendor_mismatch_idx)

    # 1× additional AMOUNT_MISMATCH (third one)
    extra_amt_idx = rng.choice([
        i for i in range(12)
        if i not in used_indices
    ])
    inv = invoices[extra_amt_idx]
    po = purchase_orders[extra_amt_idx]
    factor = 1 + rng.uniform(0.10, 0.25) * rng.choice([1, -1])
    new_amount = round(po.approved_amount * factor, 2)
    invoices[extra_amt_idx] = inv.model_copy(update={"amount": new_amount})
    planted.append(PlantedDiscrepancy(
        invoice_id=inv.invoice_id,
        po_id=po.po_id,
        discrepancy_type=DiscrepancyType.AMOUNT_MISMATCH,
        description=(
            f"Invoice {inv.invoice_id} amount ${new_amount:.2f} differs from "
            f"PO {po.po_id} approved amount ${po.approved_amount:.2f} "
            f"(off by {abs(factor - 1) * 100:.1f}%)"
        ),
    ))
    used_indices.add(extra_amt_idx)

    # 1× additional DATE_ANOMALY (second one)
    extra_date_idx = rng.choice([
        i for i in range(12)
        if i not in used_indices
    ])
    inv = invoices[extra_date_idx]
    po = purchase_orders[extra_date_idx]
    bad_date = po.approval_date - timedelta(days=rng.randint(10, 60))
    invoices[extra_date_idx] = inv.model_copy(update={"date": bad_date})
    planted.append(PlantedDiscrepancy(
        invoice_id=inv.invoice_id,
        po_id=po.po_id,
        discrepancy_type=DiscrepancyType.DATE_ANOMALY,
        description=(
            f"Invoice {inv.invoice_id} dated {bad_date.isoformat()} is before "
            f"PO {po.po_id} approval date {po.approval_date.isoformat()}"
        ),
    ))
    used_indices.add(extra_date_idx)

    # ------------------------------------------------------------------
    # 4b. Red herring invoices — look suspicious but are NOT discrepancies
    # ------------------------------------------------------------------
    # Red herring 1: invoice amount within 5% of PO (valid variance)
    rh_amt_idx = rng.choice([
        i for i in range(12)
        if i not in used_indices
    ])
    inv = invoices[rh_amt_idx]
    po = purchase_orders[rh_amt_idx]
    # Small variance: 1-4% off — suspicious looking but valid
    rh_factor = 1 + rng.uniform(0.01, 0.04) * rng.choice([1, -1])
    rh_amount = round(po.approved_amount * rh_factor, 2)
    invoices[rh_amt_idx] = inv.model_copy(update={"amount": rh_amount})
    used_indices.add(rh_amt_idx)

    # Red herring 2: invoice date same day as PO approval (edge case, valid)
    rh_date_idx = rng.choice([
        i for i in range(12)
        if i not in used_indices
    ])
    inv = invoices[rh_date_idx]
    po = purchase_orders[rh_date_idx]
    invoices[rh_date_idx] = inv.model_copy(update={"date": po.approval_date})
    used_indices.add(rh_date_idx)

    # ------------------------------------------------------------------
    # 4c. Compound discrepancy — INV-013 / PO-011 has TWO types
    # ------------------------------------------------------------------
    # Link INV-013 (index 12, originally no PO) to PO-011 (index 10)
    compound_inv_idx = 12   # INV-013
    compound_po_idx = 10    # PO-011
    compound_po = purchase_orders[compound_po_idx]
    compound_inv = invoices[compound_inv_idx]

    # Mutate: amount mismatch (~18% off)
    compound_factor = 1 + rng.uniform(0.14, 0.22)
    compound_amount = round(compound_po.approved_amount * compound_factor, 2)
    # Mutate: date anomaly (invoice dated before PO approval)
    compound_bad_date = compound_po.approval_date - timedelta(days=rng.randint(15, 45))

    invoices[compound_inv_idx] = compound_inv.model_copy(update={
        "po_number": compound_po.po_id,
        "vendor_name": compound_po.vendor_name,
        "amount": compound_amount,
        "date": compound_bad_date,
    })

    planted.append(PlantedDiscrepancy(
        invoice_id=f"INV-{compound_inv_idx+1:03d}",
        po_id=compound_po.po_id,
        discrepancy_type=DiscrepancyType.AMOUNT_MISMATCH,
        description=(
            f"Invoice INV-{compound_inv_idx+1:03d} amount ${compound_amount:.2f} differs from "
            f"PO {compound_po.po_id} approved amount ${compound_po.approved_amount:.2f}"
        ),
    ))
    planted.append(PlantedDiscrepancy(
        invoice_id=f"INV-{compound_inv_idx+1:03d}",
        po_id=compound_po.po_id,
        discrepancy_type=DiscrepancyType.DATE_ANOMALY,
        description=(
            f"Invoice INV-{compound_inv_idx+1:03d} dated {compound_bad_date.isoformat()} is before "
            f"PO {compound_po.po_id} approval date {compound_po.approval_date.isoformat()}"
        ),
    ))
    used_indices.add(compound_inv_idx)

    # ------------------------------------------------------------------
    # 5. Tickets (10 tickets)
    # ------------------------------------------------------------------
    tier_config = [
        (CustomerTier.ENTERPRISE, 4, 2),   # tier, SLA hours, count
        (CustomerTier.PRO, 8, 4),
        (CustomerTier.FREE, 24, 4),
    ]

    shared_invoice_id = invoices[rng.randint(0, 14)].invoice_id

    tickets: List[SupportTicket] = []
    ticket_ground_truths: Dict[str, TicketGroundTruth] = {}
    teams_cycle = _TICKET_TEAMS * 3  # enough for 10 tickets
    ticket_idx = 0
    near_sla_count = 0
    shared_injected = False

    for tier, sla_hours, count in tier_config:
        for _ in range(count):
            tmpl = _TICKET_TEMPLATES[ticket_idx % len(_TICKET_TEMPLATES)]

            # Inject shared_invoice_id into the first ticket whose
            # template contains {inv_ref} — this guarantees the cross-
            # task reference actually appears in the description text.
            inv_ref = f"INV-{rng.randint(1, 15):03d}"
            if not shared_injected and "{inv_ref}" in tmpl:
                inv_ref = shared_invoice_id
                shared_injected = True

            desc = tmpl.format(
                product=fake.catch_phrase(),
                tier=tier.value,
                acct=f"ACCT-{fake.random_number(digits=6, fix_len=True)}",
                inv_ref=inv_ref,
            )

            team = teams_cycle[ticket_idx]
            should_escalate = tier == CustomerTier.ENTERPRISE or rng.random() < 0.3

            # Make 2 tickets near SLA breach
            is_near_breach = False
            if near_sla_count < 2 and ticket_idx >= 2:
                created = now - timedelta(hours=sla_hours) + timedelta(minutes=rng.randint(10, 25))
                near_sla_count += 1
                is_near_breach = True
            else:
                created = now - timedelta(hours=rng.randint(1, max(1, sla_hours - 2)))

            sla_breach_at = created + timedelta(hours=sla_hours)

            # Near-SLA-breach tickets must always be escalated
            if is_near_breach:
                should_escalate = True

            ticket_id = f"TKT-{ticket_idx+1:03d}"

            tickets.append(SupportTicket(
                ticket_id=ticket_id,
                description=desc,
                customer_tier=tier,
                created_at=created,
                unresolved=True,
                sla_breach_at=sla_breach_at,
            ))

            ticket_ground_truths[ticket_id] = TicketGroundTruth(
                team=team,
                escalate=should_escalate,
                customer_tier=tier,
                sla_breach_at=sla_breach_at,
            )
            ticket_idx += 1

    rng.shuffle(tickets)

    return {
        "emails": emails,
        "tickets": tickets,
        "invoices": invoices,
        "purchase_orders": purchase_orders,
        "planted_discrepancies": planted,
        "shared_invoice_id": shared_invoice_id,
        "email_ground_truths": email_ground_truths,
        "ticket_ground_truths": ticket_ground_truths,
    }
