"""Comprehensive test suite for the InboxOps environment.

Validates:
1. reset() returns valid Observation with all data lists populated
2. step() with valid LabelEmailAction returns correct tuple
3. step() with invalid action type raises ValueError
4. Loop detection triggers -0.2 penalty after 3 identical actions
5. Task advancement after all emails are labelled
6. Seeded reproducibility
7. SQL query via QueryDatabaseAction
8. SubmitReportAction with all planted discrepancies scores ≥ 0.9
"""

from __future__ import annotations

import pytest

from environment.env import InboxOpsEnv
from environment.models import (
    FlagDiscrepancyAction,
    LabelEmailAction,
    Observation,
    QueryDatabaseAction,
    RouteTicketAction,
    StepReward,
    SubmitReportAction,
)


@pytest.fixture
def env() -> InboxOpsEnv:
    """Create a fresh environment with seed=42."""
    e = InboxOpsEnv(seed=42)
    e.reset()
    return e


# -----------------------------------------------------------------------
# Test 1: reset() returns valid Observation
# -----------------------------------------------------------------------

class TestReset:
    def test_returns_observation(self, env: InboxOpsEnv):
        obs = env.state()
        assert isinstance(obs, Observation)

    def test_inbox_populated(self, env: InboxOpsEnv):
        obs = env.state()
        assert len(obs.inbox) == 25

    def test_tickets_populated(self, env: InboxOpsEnv):
        obs = env.state()
        assert len(obs.tickets) == 10

    def test_invoices_populated(self, env: InboxOpsEnv):
        obs = env.state()
        assert len(obs.invoices) == 15

    def test_purchase_orders_populated(self, env: InboxOpsEnv):
        obs = env.state()
        assert len(obs.db_records) == 12

    def test_initial_task_is_task1(self, env: InboxOpsEnv):
        obs = env.state()
        assert obs.current_task_id == "task1"

    def test_step_count_is_zero(self, env: InboxOpsEnv):
        obs = env.state()
        assert obs.step_count == 0

    def test_task_not_complete(self, env: InboxOpsEnv):
        obs = env.state()
        assert obs.task_complete is False


# -----------------------------------------------------------------------
# Test 2: step() with valid LabelEmailAction
# -----------------------------------------------------------------------

class TestStep:
    def test_step_returns_correct_types(self, env: InboxOpsEnv):
        obs = env.state()
        email = obs.inbox[0]
        gt = env._email_gt[email.email_id]
        action = LabelEmailAction(
            email_id=email.email_id,
            label=gt.label,
            urgency=gt.urgency,
            next_action=gt.next_action,
        )
        result = env.step(action)
        assert len(result) == 4

        new_obs, reward, done, info = result
        assert isinstance(new_obs, Observation)
        assert isinstance(reward, StepReward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_perfect_email_label_scores_max(self, env: InboxOpsEnv):
        obs = env.state()
        email = obs.inbox[0]
        gt = env._email_gt[email.email_id]
        action = LabelEmailAction(
            email_id=email.email_id,
            label=gt.label,
            urgency=gt.urgency,
            next_action=gt.next_action,
        )
        _, reward, _, _ = env.step(action)
        assert reward.value == pytest.approx(0.10, abs=0.001)

    def test_step_increments_step_count(self, env: InboxOpsEnv):
        obs = env.state()
        email = obs.inbox[0]
        action = LabelEmailAction(
            email_id=email.email_id,
            label="general",
            urgency=1,
            next_action="reply",
        )
        new_obs, _, _, _ = env.step(action)
        assert new_obs.step_count == 1


# -----------------------------------------------------------------------
# Test 3: step() with invalid action raises ValueError
# -----------------------------------------------------------------------

class TestInvalidAction:
    def test_invalid_action_type_raises(self, env: InboxOpsEnv):
        with pytest.raises((ValueError, TypeError)):
            env.step("not_an_action")  # type: ignore[arg-type]

    def test_invalid_email_id_returns_penalty(self, env: InboxOpsEnv):
        action = LabelEmailAction(
            email_id="NONEXISTENT-999",
            label="general",
            urgency=1,
            next_action="reply",
        )
        _, reward, _, _ = env.step(action)
        assert reward.value < 0


# -----------------------------------------------------------------------
# Test 4: Loop detection
# -----------------------------------------------------------------------

class TestLoopDetection:
    def test_third_identical_action_triggers_penalty(self, env: InboxOpsEnv):
        obs = env.state()
        email = obs.inbox[0]
        action = LabelEmailAction(
            email_id=email.email_id,
            label="general",
            urgency=1,
            next_action="reply",
        )
        # First two should not trigger loop
        _, r1, _, _ = env.step(action)
        _, r2, _, _ = env.step(action)
        assert r1.value != -0.2 or "loop_detected" not in r1.info.get("reason", "")
        assert r2.value != -0.2 or "loop_detected" not in r2.info.get("reason", "")

        # Third should trigger loop detection
        _, r3, _, _ = env.step(action)
        assert r3.value == pytest.approx(-0.2)
        assert r3.info.get("reason") == "loop_detected"


# -----------------------------------------------------------------------
# Test 5: Task advancement
# -----------------------------------------------------------------------

class TestTaskAdvancement:
    def test_advance_to_task2_after_all_emails(self, env: InboxOpsEnv):
        obs = env.state()
        assert obs.current_task_id == "task1"

        # Label all 25 emails
        for email in obs.inbox:
            gt = env._email_gt[email.email_id]
            action = LabelEmailAction(
                email_id=email.email_id,
                label=gt.label,
                urgency=gt.urgency,
                next_action=gt.next_action,
            )
            new_obs, _, _, _ = env.step(action)

        assert new_obs.current_task_id == "task2"

    def test_advance_to_task3_after_all_tickets(self, env: InboxOpsEnv):
        obs = env.state()

        # Complete task 1
        for email in obs.inbox:
            gt = env._email_gt[email.email_id]
            action = LabelEmailAction(
                email_id=email.email_id,
                label=gt.label,
                urgency=gt.urgency,
                next_action=gt.next_action,
            )
            obs, _, _, _ = env.step(action)

        assert obs.current_task_id == "task2"

        # Complete task 2
        for ticket in obs.tickets:
            gt = env._ticket_gt[ticket.ticket_id]
            action = RouteTicketAction(
                ticket_id=ticket.ticket_id,
                team=gt.team,
                escalate=gt.escalate,
                draft_message="Routing to appropriate team for billing invoice account review.",
            )
            obs, _, _, _ = env.step(action)

        assert obs.current_task_id == "task3"


# -----------------------------------------------------------------------
# Test 6: Seeded reproducibility
# -----------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_identical_episodes(self):
        env1 = InboxOpsEnv(seed=42)
        obs1 = env1.reset()

        env2 = InboxOpsEnv(seed=42)
        obs2 = env2.reset()

        # Compare emails
        assert len(obs1.inbox) == len(obs2.inbox)
        for e1, e2 in zip(obs1.inbox, obs2.inbox):
            assert e1.email_id == e2.email_id
            assert e1.subject == e2.subject
            assert env1._email_gt[e1.email_id].label == env2._email_gt[e2.email_id].label

        # Compare tickets
        assert len(obs1.tickets) == len(obs2.tickets)
        for t1, t2 in zip(obs1.tickets, obs2.tickets):
            assert t1.ticket_id == t2.ticket_id
            assert t1.customer_tier == t2.customer_tier

        # Compare invoices
        assert len(obs1.invoices) == len(obs2.invoices)
        for i1, i2 in zip(obs1.invoices, obs2.invoices):
            assert i1.invoice_id == i2.invoice_id
            assert i1.amount == i2.amount

    def test_different_seeds_produce_different_episodes(self):
        env1 = InboxOpsEnv(seed=42)
        obs1 = env1.reset()

        env2 = InboxOpsEnv(seed=99)
        obs2 = env2.reset()

        # At least some emails should differ in subject
        subjects1 = {e.subject for e in obs1.inbox}
        subjects2 = {e.subject for e in obs2.inbox}
        assert subjects1 != subjects2


# -----------------------------------------------------------------------
# Test 7: SQL query via QueryDatabaseAction
# -----------------------------------------------------------------------

class TestQueryDB:
    def test_query_returns_rows(self, env: InboxOpsEnv):
        # Advance to task 3 to ensure SQL is valid context
        action = QueryDatabaseAction(sql="SELECT COUNT(*) FROM purchase_orders")
        _, reward, _, info = env.step(action)
        result = info.get("query_result", {})
        assert result.get("error") is None
        assert len(result.get("rows", [])) == 1
        assert result["rows"][0][0] == 12

    def test_query_invoices_table(self, env: InboxOpsEnv):
        action = QueryDatabaseAction(sql="SELECT COUNT(*) FROM invoices")
        _, _, _, info = env.step(action)
        result = info["query_result"]
        assert result["error"] is None
        assert result["rows"][0][0] == 15

    def test_invalid_sql_returns_error(self, env: InboxOpsEnv):
        action = QueryDatabaseAction(sql="SELECT * FROM nonexistent_table")
        _, _, _, info = env.step(action)
        result = info["query_result"]
        assert result["error"] is not None

    def test_join_query(self, env: InboxOpsEnv):
        action = QueryDatabaseAction(
            sql=(
                "SELECT i.invoice_id, i.amount, p.approved_amount "
                "FROM invoices i "
                "JOIN purchase_orders p ON i.po_number = p.po_id "
                "WHERE ABS(i.amount - p.approved_amount) / p.approved_amount > 0.05"
            )
        )
        _, _, _, info = env.step(action)
        result = info["query_result"]
        assert result["error"] is None
        # Should find at least the 2 planted amount mismatches
        assert len(result["rows"]) >= 2


# -----------------------------------------------------------------------
# Test 8: SubmitReportAction with all planted discrepancies
# -----------------------------------------------------------------------

class TestSubmitReport:
    def test_perfect_report_scores_high(self, env: InboxOpsEnv):
        # Get planted discrepancies
        planted = env._episode["planted_discrepancies"]
        assert len(planted) == 8

        # Flag all planted discrepancies
        for p in planted:
            flag = FlagDiscrepancyAction(
                invoice_id=p.invoice_id,
                po_id=p.po_id,
                discrepancy_type=p.discrepancy_type.value,
                explanation=p.description,
            )
            env.step(flag)

        # Submit report
        report_action = SubmitReportAction(
            report={
                "discrepancies": [
                    {
                        "invoice_id": p.invoice_id,
                        "type": p.discrepancy_type.value,
                        "description": p.description,
                    }
                    for p in planted
                ],
                "summary": "Found all 8 discrepancies.",
            }
        )
        _, reward, done, info = env.step(report_action)

        assert done is True
        assert reward.value >= 0.9
        assert "final_scores" in info

    def test_empty_report_scores_zero(self):
        env = InboxOpsEnv(seed=42)
        env.reset()

        report_action = SubmitReportAction(report={"discrepancies": [], "summary": "None found."})
        _, reward, done, _ = env.step(report_action)

        assert done is True
        # No flags → no score (but no false positives either)
        assert reward.value == pytest.approx(0.0)

    def test_false_positives_penalised(self):
        env = InboxOpsEnv(seed=42)
        env.reset()

        # Flag something that isn't planted
        false_flag = FlagDiscrepancyAction(
            invoice_id="INV-999",
            po_id="PO-999",
            discrepancy_type="amount_mismatch",
            explanation="Made up discrepancy",
        )
        env.step(false_flag)

        report_action = SubmitReportAction(report={"discrepancies": [], "summary": "One issue."})
        _, reward, done, _ = env.step(report_action)

        assert done is True
        # Should be negative or zero due to false positive
        assert reward.value <= 0.0


# -----------------------------------------------------------------------
# Integration: Full episode
# -----------------------------------------------------------------------

class TestFullEpisode:
    def test_complete_episode_with_perfect_actions(self):
        env = InboxOpsEnv(seed=42)
        obs = env.reset()

        # Task 1: Label all emails perfectly
        for email in obs.inbox:
            gt = env._email_gt[email.email_id]
            action = LabelEmailAction(
                email_id=email.email_id,
                label=gt.label,
                urgency=gt.urgency,
                next_action=gt.next_action,
            )
            obs, _, _, _ = env.step(action)

        assert obs.current_task_id == "task2"

        # Task 2: Route all tickets perfectly
        for ticket in obs.tickets:
            gt = env._ticket_gt[ticket.ticket_id]
            action = RouteTicketAction(
                ticket_id=ticket.ticket_id,
                team=gt.team,
                escalate=gt.escalate,
                draft_message="Routing for billing invoice account pipeline review.",
            )
            obs, _, _, _ = env.step(action)

        assert obs.current_task_id == "task3"

        # Task 3: Flag all planted discrepancies
        planted = env._episode["planted_discrepancies"]
        for p in planted:
            flag = FlagDiscrepancyAction(
                invoice_id=p.invoice_id,
                po_id=p.po_id,
                discrepancy_type=p.discrepancy_type.value,
                explanation=p.description,
            )
            env.step(flag)

        report = SubmitReportAction(report={"summary": "All discrepancies found."})
        obs, reward, done, info = env.step(report)

        assert done is True
        assert "final_scores" in info
        # Perfect play should yield high scores
        assert info["final_scores"]["task1"] >= 0.9
        assert info["final_scores"]["task3"] >= 0.9


# -----------------------------------------------------------------------
# Test 11: All incremental StepReward.value within [-0.2, 0.2]
# -----------------------------------------------------------------------

class TestRewardRange:
    def test_email_rewards_within_range(self, env: InboxOpsEnv):
        """All email labelling step rewards must be in [-0.2, 0.2]."""
        obs = env.state()
        for email in obs.inbox:
            gt = env._email_gt[email.email_id]
            action = LabelEmailAction(
                email_id=email.email_id,
                label=gt.label,
                urgency=gt.urgency,
                next_action=gt.next_action,
            )
            _, reward, done, _ = env.step(action)
            if not done:
                assert -0.2 <= reward.value <= 0.2, (
                    f"Reward {reward.value} out of [-0.2, 0.2] for {email.email_id}"
                )

    def test_ticket_rewards_within_range(self):
        """All ticket routing step rewards must be in [-0.2, 0.2]."""
        env = InboxOpsEnv(seed=42)
        obs = env.reset()

        # Complete task 1 first
        for email in obs.inbox:
            gt = env._email_gt[email.email_id]
            action = LabelEmailAction(
                email_id=email.email_id,
                label=gt.label,
                urgency=gt.urgency,
                next_action=gt.next_action,
            )
            obs, _, _, _ = env.step(action)

        # Now check ticket rewards
        for ticket in obs.tickets:
            gt = env._ticket_gt[ticket.ticket_id]
            action = RouteTicketAction(
                ticket_id=ticket.ticket_id,
                team=gt.team,
                escalate=gt.escalate,
                draft_message="Routing for billing invoice account pipeline review.",
            )
            _, reward, done, _ = env.step(action)
            if not done:
                assert -0.2 <= reward.value <= 0.2, (
                    f"Reward {reward.value} out of [-0.2, 0.2] for {ticket.ticket_id}"
                )

    def test_loop_penalty_within_range(self, env: InboxOpsEnv):
        """Loop penalty of -0.2 must be at the boundary of [-0.2, 0.2]."""
        obs = env.state()
        email = obs.inbox[0]
        action = LabelEmailAction(
            email_id=email.email_id,
            label="general",
            urgency=1,
            next_action="reply",
        )
        env.step(action)
        env.step(action)
        _, reward, _, _ = env.step(action)
        assert -0.2 <= reward.value <= 0.2


# -----------------------------------------------------------------------
# Test 12: info["final_scores"] present and all values in [0.0, 1.0]
# -----------------------------------------------------------------------

class TestFinalScores:
    def test_final_scores_present_when_done(self):
        """info must contain final_scores with all values in [0.0, 1.0] when done."""
        env = InboxOpsEnv(seed=42)
        obs = env.reset()

        # Task 1
        for email in obs.inbox:
            gt = env._email_gt[email.email_id]
            action = LabelEmailAction(
                email_id=email.email_id,
                label=gt.label,
                urgency=gt.urgency,
                next_action=gt.next_action,
            )
            obs, _, _, _ = env.step(action)

        # Task 2
        for ticket in obs.tickets:
            gt = env._ticket_gt[ticket.ticket_id]
            action = RouteTicketAction(
                ticket_id=ticket.ticket_id,
                team=gt.team,
                escalate=gt.escalate,
                draft_message="Routing for billing invoice account pipeline review.",
            )
            obs, _, _, _ = env.step(action)

        # Task 3 — submit report
        report = SubmitReportAction(report={"summary": "Done."})
        _, _, done, info = env.step(report)

        assert done is True
        assert "final_scores" in info
        assert "episode_complete" in info
        assert info["episode_complete"] is True

        for task_id in ("task1", "task2", "task3"):
            score = info["final_scores"][task_id]
            assert 0.0 <= score <= 1.0, (
                f"final_scores[{task_id}] = {score} is outside [0.0, 1.0]"
            )


# -----------------------------------------------------------------------
# Test: state() returns current observation
# -----------------------------------------------------------------------

class TestStateMethod:
    def test_state_returns_current_observation(self):
        env = InboxOpsEnv(seed=42)
        obs = env.reset()
        state = env.state()
        assert state.current_task_id == obs.current_task_id
        assert state.step_count == obs.step_count
