"""InboxOps inference script."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import Any

from openai import OpenAI

from environment.env import InboxOpsEnv
from environment.models import (
    Action,
    FlagDiscrepancyAction,
    LabelEmailAction,
    QueryDatabaseAction,
    RouteTicketAction,
    SubmitReportAction,
)

# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------
def _timeout_handler(sig, frame):
    print("[DEBUG] timeout", file=sys.stderr, flush=True)
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(18 * 60)

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "default-model"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )
except Exception as e:
    print(f"[DEBUG] Failed to initialize OpenAI client: {e}", file=sys.stderr, flush=True)
    client = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS = 200

SYSTEM_PROMPT = """
You are an operations analyst. You will be given an inbox, ticket queue,
and financial records. Respond ONLY with a valid JSON action object.

For Task 1 (email triage), respond with:
{"action_type": "label_email", "email_id": "...", "label": "billing|onboarding|outage|spam|general", "urgency": 1-5, "next_action": "reply|escalate|archive|forward"}

For Task 2 (ticket routing), respond with:
{"action_type": "route_ticket", "ticket_id": "...", "team": "billing|infra|product|account_management", "escalate": true|false, "draft_message": "..."}

For Task 3 (reconciliation), first use query_db to investigate, then flag discrepancies, then submit report:
{"action_type": "query_db", "sql": "SELECT ..."}
{"action_type": "flag_discrepancy", "invoice_id": "...", "po_id": "...", "discrepancy_type": "amount_mismatch|duplicate_line_item|missing_po|date_anomaly|vendor_mismatch", "explanation": "..."}
{"action_type": "submit_report", "report": {"discrepancies": [...], "summary": "..."}}

Always process urgent items first. Check SLA breach times carefully.
Cross-reference invoice IDs mentioned in ticket descriptions with Task 3 data.
""".strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _debug(*args, **kwargs) -> None:
    print("[DEBUG]", *args, file=sys.stderr, flush=True, **kwargs)


def build_context(obs: Any) -> str:
    lines: list[str] = []
    lines.append(f"=== Current Task: {obs.current_task_id} (step {obs.step_count}) ===\n")

    if obs.current_task_id == "task1":
        lines.append("## Inbox (process each email)\n")
        for email in obs.inbox:
            d = email.model_dump()
            d["timestamp"] = d["timestamp"].isoformat() if hasattr(d["timestamp"], "isoformat") else str(d["timestamp"])
            lines.append(json.dumps(d, indent=2))
            lines.append("")
    elif obs.current_task_id == "task2":
        lines.append("## Ticket Queue (route each ticket)\n")
        for ticket in obs.tickets:
            d = ticket.model_dump()
            for k in ("created_at", "sla_breach_at"):
                if k in d and hasattr(d[k], "isoformat"):
                    d[k] = d[k].isoformat()
            lines.append(json.dumps(d, indent=2))
            lines.append("")
    elif obs.current_task_id == "task3":
        lines.append("## Financial Records\n")
        lines.append("### Invoices\n")
        for inv in obs.invoices:
            d = inv.model_dump()
            if hasattr(d.get("date"), "isoformat"):
                d["date"] = d["date"].isoformat()
            lines.append(json.dumps(d, indent=2))
        lines.append(f"\n{len(obs.invoices)} total invoices")
        lines.append("\n### Purchase Orders (query via SQL)")
        lines.append("Tables: purchase_orders(po_id, vendor_name, approved_amount, approval_date, status)")
        lines.append("        invoices(invoice_id, vendor_name, amount, date, po_number)\n")
        lines.append("Use query_db actions to investigate, then flag_discrepancy, then submit_report.")

    return "\n".join(lines)


def parse_action(raw: str) -> Action | None:
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        data = json.loads(text)
        action_type = data.get("action_type", "")

        if action_type == "label_email":
            return LabelEmailAction(**data)
        elif action_type == "route_ticket":
            return RouteTicketAction(**data)
        elif action_type == "query_db":
            return QueryDatabaseAction(**data)
        elif action_type == "flag_discrepancy":
            return FlagDiscrepancyAction(**data)
        elif action_type == "submit_report":
            return SubmitReportAction(**data)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
    except json.JSONDecodeError:
        _debug("Failed to decode JSON from model block.")
        return Action(action_type="label_email", email_id="unknown", label="general", urgency=1, next_action="archive")
    except Exception as e:
        _debug(f"Action parsing error: {e}")
        return Action(action_type="label_email", email_id="unknown", label="general", urgency=1, next_action="archive")


def run_task(task_name: str, seed: int):
    _debug(f"Initializing task: {task_name} with seed {seed}")
    
    env = None
    step_count = 0
    total_reward = 0.0
    rewards_list = []
    success = False
    # Will be set from env.max_reward after reset(); stays 0 on early failure
    max_total_reward = 0.0

    def log_end(success_flag, s_count, raw_score_val, r_list):
        str_success = "true" if success_flag else "false"
        epsilon = 1e-6
        safe_score = min(max(raw_score_val, epsilon), 1.0 - epsilon)
        rewards_str = ",".join([f"{r:.2f}" for r in r_list])
        print(f"[END] success={str_success} steps={s_count} score={safe_score:.2f} rewards={rewards_str}", flush=True)

    try:
        try:
            env = InboxOpsEnv(seed=seed)
        except Exception as e:
            print(f"[DEBUG] Failed to initialize environment: {e}", flush=True)
            log_end(success=False, s_count=0, raw_score_val=0.0, r_list=[])
            return

        try:
            obs = env.reset()
            # Read the true theoretical max from the environment so
            # score normalisation stays in sync with the generator.
            max_total_reward = env.max_reward
        except Exception as e:
            print(f"[DEBUG] env.reset() failed: {e}", flush=True)
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
            log_end(success=False, s_count=0, raw_score_val=0.0, r_list=[])
            return

        print(f"[START] task={task_name} env=InboxOpsEnv model={MODEL_NAME}", flush=True)

        while step_count < MAX_STEPS:
            context = build_context(obs)
            error_msg = "null"

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": context},
                    ],
                    max_tokens=256,
                )
                raw = response.choices[0].message.content or "hello"
                if not raw.strip():
                    raw = "hello"
            except Exception as exc:
                _debug(f"API error: {exc}")
                raw = "hello"
                error_msg = str(exc).replace(" ", "_")

            action = parse_action(raw)
            if action is None:
                step_count += 1
                continue

            try:
                obs, reward, done, info = env.step(action)
            except Exception as exc:
                print(f"[DEBUG] env.step() failed: {exc}", flush=True)
                error_msg = str(exc).replace(" ", "_")
                val = 0.0
                total_reward += val
                rewards_list.append(val)
                print(f"[STEP] step={step_count + 1} action={action.action_type} reward={val:.2f} done=true error={error_msg}", flush=True)
                break

            val = float(reward.value)
            total_reward += val
            rewards_list.append(val)
            
            str_done = "true" if done else "false"
            print(f"[STEP] step={step_count + 1} action={action.action_type} reward={val:.2f} done={str_done} error={error_msg}", flush=True)

            if done:
                success = True
                break

            step_count += 1

    except Exception as e:
        _debug(f"Execution error: {e}")
    finally:
        total_sum = sum(rewards_list)
        raw_score = total_sum / max_total_reward if max_total_reward > 0 else 0.0
        
        # No artificial fallback — failed runs get honest 0.0

        log_end(success, step_count, raw_score, rewards_list)
        
        if env and hasattr(env, "close"):
            try:
                env.close()
            except Exception as ce:
                _debug(f"Error closing env: {ce}")


def main():
    if client is None:
        _debug("OpenAI client not initialized, exiting.")
        sys.exit(1)

    # 1. Multi-Task Execution: Loop > 3 times
    TASKS = ["InboxOps_Easy", "InboxOps_Medium", "InboxOps_Hard"]
    
    base_seed = int(os.getenv("SEED", 42))
    for idx, task_name in enumerate(TASKS):
        run_task(task_name, seed=base_seed + idx)

if __name__ == "__main__":
    main()
