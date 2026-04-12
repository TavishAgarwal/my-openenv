"""Tests for reconciliation grader — vendor_mismatch and red herring scoring."""

from __future__ import annotations

import pytest

from environment.env import InboxOpsEnv
from environment.models import (
    FlagDiscrepancyAction,
    SubmitReportAction,
)


class TestVendorMismatchScoring:
    def test_vendor_mismatch_scores_full(self):
        """Flag a vendor_mismatch discrepancy with correct type → 0.20 credit."""
        env = InboxOpsEnv(seed=42)
        env.reset()

        planted = env._episode["planted_discrepancies"]
        # Find the vendor_mismatch discrepancy
        vm = [p for p in planted if p.discrepancy_type.value == "vendor_mismatch"]
        assert len(vm) >= 1, "Expected at least one vendor_mismatch planted"

        target = vm[0]
        flag = FlagDiscrepancyAction(
            invoice_id=target.invoice_id,
            po_id=target.po_id,
            discrepancy_type="vendor_mismatch",
            explanation="Vendor name on invoice does not match PO vendor.",
        )
        env.step(flag)

        report = SubmitReportAction(report={"summary": "One vendor mismatch found."})
        _, reward, done, info = env.step(report)

        assert done is True
        # One correct flag out of 8 planted: raw = 0.20, max = 8*0.20 = 1.60
        # normalized = 0.20 / 1.60 = 0.125
        expected = 0.20 / (len(planted) * 0.20)
        assert reward.value == pytest.approx(expected, abs=0.01)


class TestRedHerringPenalised:
    def test_red_herring_penalised(self):
        """Flag an invoice that is NOT in planted_discrepancies → −0.10."""
        env = InboxOpsEnv(seed=42)
        env.reset()

        planted = env._episode["planted_discrepancies"]
        planted_invoice_ids = {p.invoice_id for p in planted}

        # Find an invoice that is NOT planted
        clean_invoice = None
        for inv in env._state.invoices:
            if inv.invoice_id not in planted_invoice_ids:
                clean_invoice = inv
                break
        assert clean_invoice is not None, "Expected at least one clean invoice"

        false_flag = FlagDiscrepancyAction(
            invoice_id=clean_invoice.invoice_id,
            po_id=clean_invoice.po_number or "PO-NONE",
            discrepancy_type="amount_mismatch",
            explanation="Flagging a clean invoice.",
        )
        env.step(false_flag)

        report = SubmitReportAction(report={"summary": "Flagged one."})
        _, reward, done, info = env.step(report)

        assert done is True
        # Only a false positive: raw = -0.10, max = 8*0.20 = 1.60
        # normalized = max(0.0, -0.10/1.60) = 0.0  (clamped)
        assert reward.value == pytest.approx(0.0, abs=0.01)
        # Verify the false positive appears in the breakdown
        assert any("false_positive" in k for k in reward.breakdown)
