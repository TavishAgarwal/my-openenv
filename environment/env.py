"""InboxOpsEnv — the core environment class.

Implements the OpenEnv contract: reset() → Observation, step(Action) → (Observation, StepReward, bool, dict).
Manages task sequencing (email → tickets → reconciliation), loop detection,
and an in-memory SQLite database for the reconciliation task.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import datetime
from typing import List

from environment.generator import generate_episode
from environment.graders.email_grader import grade_email_action
from environment.graders.reconciliation_grader import (
    grade_query_action,
    grade_report_submission,
)
from environment.graders.ticket_grader import grade_ticket_action
from environment.models import (
    Action,
    EmailGroundTruth,
    FlagDiscrepancyAction,
    LabelEmailAction,
    Observation,
    PlantedDiscrepancy,
    QueryDatabaseAction,
    RouteTicketAction,
    StepReward,
    SubmitReportAction,
    TicketGroundTruth,
)


# Score epsilon — ensures task scores are strictly in (0, 1)
_SCORE_EPS = 1e-6


class InboxOpsEnv:
    """OpenEnv-compliant environment simulating an operations analyst's workload."""

    # Fixed reference time matching the generator
    _NOW = datetime(2025, 3, 15, 9, 0, 0)

    # Step-waste penalty configuration
    MAX_STEPS: int = 200                # total step budget per episode
    STEP_DECAY_THRESHOLD: int = 50      # step at which decay begins

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._episode: dict | None = None
        self._state: Observation | None = None
        self._action_history: Counter = Counter()
        self._scores: dict[str, float] = {"task1": 0.0, "task2": 0.0, "task3": 0.0}
        self._sqlite_conn: sqlite3.Connection | None = None
        self._labeled_emails: dict[str, LabelEmailAction] = {}
        self._routed_tickets: dict[str, RouteTicketAction] = {}
        self._flagged: List[FlagDiscrepancyAction] = []
        # Ground-truth stores — populated during reset(), never exposed
        self._email_gt: dict[str, EmailGroundTruth] = {}
        self._ticket_gt: dict[str, TicketGroundTruth] = {}
        # Previous-action tracking for repeat-action penalty
        self._prev_action_key: tuple | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Regenerate episode, reset all state, return initial Observation."""
        self._episode = generate_episode(self.seed)
        self._state = Observation(
            inbox=self._episode["emails"],
            tickets=self._episode["tickets"],
            db_records=self._episode["purchase_orders"],
            invoices=self._episode["invoices"],
            current_task_id="task1",
            step_count=0,
            task_complete=False,
        )
        self._init_sqlite()
        self._action_history = Counter()
        self._labeled_emails = {}
        self._routed_tickets = {}
        self._flagged = []
        self._scores = {"task1": 0.0, "task2": 0.0, "task3": 0.0}
        self._prev_action_key = None
        # Populate private ground-truth stores from episode data
        self._email_gt = self._episode["email_ground_truths"]
        self._ticket_gt = self._episode["ticket_ground_truths"]
        return self._state

    def step(self, action: Action) -> tuple[Observation, StepReward, bool, dict]:
        """Dispatch action to the appropriate grader.

        Handles loop detection, task advancement, and score accumulation.
        Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised — call reset() first.")

        self._state = self._state.model_copy(
            update={"step_count": self._state.step_count + 1}
        )

        # --- Loop detection ---
        if self._detect_loop(action):
            reward = StepReward(
                value=-0.2,
                breakdown={"loop_penalty": -0.2},
                done=False,
                info={"reason": "loop_detected"},
            )
            return self._state, reward, False, reward.info

        # --- Dispatch by action type ---
        if isinstance(action, LabelEmailAction):
            reward = self._handle_label_email(action)
        elif isinstance(action, RouteTicketAction):
            reward = self._handle_route_ticket(action)
        elif isinstance(action, QueryDatabaseAction):
            reward = self._handle_query_db(action)
        elif isinstance(action, FlagDiscrepancyAction):
            reward = self._handle_flag_discrepancy(action)
        elif isinstance(action, SubmitReportAction):
            reward = self._handle_submit_report(action)
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

        # --- Step-waste penalties (applied before per-step clamp) ---
        if not reward.done:
            current_key = self._action_key(action)
            new_value = reward.value
            updated_breakdown = dict(reward.breakdown)

            # 1) Repeat-action penalty: −0.01 if this action is identical
            #    to the immediately preceding one (same type + key fields).
            if current_key is not None and current_key == self._prev_action_key:
                new_value -= 0.01
                updated_breakdown["repeat_action_penalty"] = -0.01

            # 2) Step-decay: after STEP_DECAY_THRESHOLD, gently reduce
            #    rewards so agents that finish faster score higher.
            #    factor = max(0.5, 1.0 − (step − threshold) / MAX_STEPS × 0.3)
            step = self._state.step_count
            if step > self.STEP_DECAY_THRESHOLD:
                factor = max(
                    0.5,
                    1.0 - (step - self.STEP_DECAY_THRESHOLD) / self.MAX_STEPS * 0.3,
                )
                new_value = new_value * factor
                updated_breakdown["step_decay_factor"] = round(factor, 4)

            if new_value != reward.value or updated_breakdown != reward.breakdown:
                reward = reward.model_copy(
                    update={"value": round(new_value, 6), "breakdown": updated_breakdown}
                )

            self._prev_action_key = current_key

        # --- Clamp per-step reward to [-0.2, 0.2] (terminal rewards exempt) ---
        if not reward.done:
            clamped_val = max(-0.2, min(0.2, reward.value))
            if clamped_val != reward.value:
                reward = reward.model_copy(update={"value": clamped_val})

        # --- Check task advancement ---
        self._advance_task()

        done = self._state.task_complete
        info = reward.info.copy()
        if done:
            clamped_scores = {
                k: min(max(v, _SCORE_EPS), 1.0 - _SCORE_EPS)
                for k, v in self._scores.items()
            }
            info["final_scores"] = clamped_scores
            info["episode_complete"] = True
            # Also clamp the reward.value itself — the API serializes this
            # directly and the validator may read it as the "task score".
            clamped_reward_val = min(max(reward.value, _SCORE_EPS), 1.0 - _SCORE_EPS)
            if clamped_reward_val != reward.value:
                reward = reward.model_copy(update={"value": clamped_reward_val})

        return self._state, reward, done, info

    def state(self) -> Observation:
        """Return current Observation (OpenEnv contract)."""
        if self._state is None:
            raise RuntimeError("Environment not initialised — call reset() first.")
        return self._state

    @property
    def max_reward(self) -> float:
        """Theoretical maximum sum of per-step rewards across all tasks.

        Computed from the actual episode data so it stays in sync with the
        generator.  The calculation assumes every action earns the best
        possible per-step reward:

          Task 1 (email triage):
            Each email can earn up to 0.10 per step
            (label=0.05 + urgency_exact=0.03 + next_action=0.02).
            Per-step clamp is [-0.2, 0.2], so 0.10 passes through.
            → n_emails × 0.10

          Task 2 (ticket routing):
            Each ticket can earn up to 0.20 per step
            (team=0.10 + escalate=0.05 + draft_keywords=0.05).
            0.20 is exactly the per-step clamp ceiling.
            → n_tickets × 0.20

          Task 3 (reconciliation):
            query_db and flag_discrepancy steps score 0.0.
            submit_report returns a normalized [0, 1] score and is
            exempt from per-step clamping (done=True).
            → 1.0
        """
        if self._episode is None:
            raise RuntimeError("Environment not initialised — call reset() first.")

        n_emails = len(self._episode["emails"])
        n_tickets = len(self._episode["tickets"])
        return n_emails * 0.10 + n_tickets * 0.20 + 1.0

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_label_email(self, action: LabelEmailAction) -> StepReward:
        gt = self._email_gt.get(action.email_id)
        if gt is None:
            return StepReward(
                value=-0.05,
                breakdown={"invalid_email_id": -0.05},
                done=False,
                info={"error": f"Email {action.email_id} not found"},
            )
        self._labeled_emails[action.email_id] = action
        reward = grade_email_action(action, gt)
        self._scores["task1"] += reward.value
        return reward

    def _handle_route_ticket(self, action: RouteTicketAction) -> StepReward:
        gt = self._ticket_gt.get(action.ticket_id)
        if gt is None:
            return StepReward(
                value=-0.05,
                breakdown={"invalid_ticket_id": -0.05},
                done=False,
                info={"error": f"Ticket {action.ticket_id} not found"},
            )
        self._routed_tickets[action.ticket_id] = action
        reward = grade_ticket_action(action, gt, self._NOW)
        self._scores["task2"] += reward.value
        return reward

    def _handle_query_db(self, action: QueryDatabaseAction) -> StepReward:
        if self._sqlite_conn is None:
            return StepReward(
                value=0.0,
                breakdown={},
                done=False,
                info={"error": "SQLite not initialised"},
            )
        result = grade_query_action(action.sql, self._sqlite_conn)
        return StepReward(
            value=0.0,  # queries are informational, no score
            breakdown={},
            done=False,
            info={"query_result": result},
        )

    def _handle_flag_discrepancy(self, action: FlagDiscrepancyAction) -> StepReward:
        self._flagged.append(action)
        return StepReward(
            value=0.0,  # score awarded at report submission
            breakdown={"flag_recorded": 0.0},
            done=False,
            info={"flagged_count": len(self._flagged)},
        )

    def _handle_submit_report(self, action: SubmitReportAction) -> StepReward:
        planted: List[PlantedDiscrepancy] = self._episode["planted_discrepancies"]  # type: ignore[index]
        reward = grade_report_submission(self._flagged, planted)
        self._scores["task3"] = min(max(reward.value, _SCORE_EPS), 1.0 - _SCORE_EPS)
        # Mark task 3 as complete
        self._state = self._state.model_copy(update={"task_complete": True})  # type: ignore[union-attr]
        return reward

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def _init_sqlite(self) -> None:
        """Create in-memory SQLite DB with purchase_orders and invoices tables."""
        conn = sqlite3.connect(":memory:")

        conn.execute("""
            CREATE TABLE purchase_orders (
                po_id TEXT PRIMARY KEY,
                vendor_name TEXT,
                approved_amount REAL,
                approval_date TEXT,
                status TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE invoices (
                invoice_id TEXT PRIMARY KEY,
                vendor_name TEXT,
                amount REAL,
                date TEXT,
                po_number TEXT
            )
        """)

        for po in self._episode["purchase_orders"]:  # type: ignore[index]
            conn.execute(
                "INSERT INTO purchase_orders VALUES (?, ?, ?, ?, ?)",
                (po.po_id, po.vendor_name, po.approved_amount,
                 po.approval_date.isoformat(), po.status),
            )

        for inv in self._episode["invoices"]:  # type: ignore[index]
            conn.execute(
                "INSERT INTO invoices VALUES (?, ?, ?, ?, ?)",
                (inv.invoice_id, inv.vendor_name, inv.amount,
                 inv.date.isoformat(), inv.po_number),
            )

        conn.commit()
        self._sqlite_conn = conn

    def _advance_task(self) -> None:
        """Move from task1 → task2 → task3 → done."""
        if self._state is None:
            return

        current = self._state.current_task_id

        if current == "task1":
            total_emails = len(self._state.inbox)
            if len(self._labeled_emails) >= total_emails:
                # Normalize task1 score
                max_possible = total_emails * 0.10
                if max_possible > 0:
                    self._scores["task1"] = min(1.0 - _SCORE_EPS, max(_SCORE_EPS, self._scores["task1"] / max_possible))
                self._state = self._state.model_copy(update={"current_task_id": "task2"})

        elif current == "task2":
            total_tickets = len(self._state.tickets)
            if len(self._routed_tickets) >= total_tickets:
                # Normalize task2 score
                max_possible = total_tickets * 0.20  # 0.10 + 0.05 + 0.05
                if max_possible > 0:
                    self._scores["task2"] = min(1.0 - _SCORE_EPS, max(_SCORE_EPS, self._scores["task2"] / max_possible))
                self._state = self._state.model_copy(update={"current_task_id": "task3"})

        # task3 completion is handled by _handle_submit_report

    def _detect_loop(self, action: Action) -> bool:
        """Return True if this exact action has been taken 3+ times."""
        if isinstance(action, LabelEmailAction):
            key = ("label_email", action.email_id)
        elif isinstance(action, RouteTicketAction):
            key = ("route_ticket", action.ticket_id)
        elif isinstance(action, QueryDatabaseAction):
            key = ("query_db", action.sql)
        elif isinstance(action, FlagDiscrepancyAction):
            key = ("flag_discrepancy", action.invoice_id)
        elif isinstance(action, SubmitReportAction):
            key = ("submit_report", "report")
        else:
            return False

        self._action_history[key] += 1
        return self._action_history[key] >= 3

    @staticmethod
    def _action_key(action: Action) -> tuple | None:
        """Return a hashable key for *action* based on type + key fields.

        Used by the repeat-action penalty to compare consecutive actions.
        Returns None for unrecognised action types (no penalty applied).
        """
        if isinstance(action, LabelEmailAction):
            return ("label_email", action.email_id, action.label, action.urgency, action.next_action)
        elif isinstance(action, RouteTicketAction):
            return ("route_ticket", action.ticket_id, action.team, action.escalate)
        elif isinstance(action, QueryDatabaseAction):
            return ("query_db", action.sql)
        elif isinstance(action, FlagDiscrepancyAction):
            return ("flag_discrepancy", action.invoice_id, action.po_id, action.discrepancy_type)
        elif isinstance(action, SubmitReportAction):
            return ("submit_report",)
        return None
