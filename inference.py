"""InboxOps inference script.

Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Uses the OpenAI client pointed at the HF Inference API.
Outputs structured [START]/[STEP]/[END] JSON logs to stdout.
All debug/info messages go to stderr.
"""

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
# Timeout guard (18-minute hard limit — 2 min buffer for 20-min constraint)
# ---------------------------------------------------------------------------
def _timeout_handler(sig, frame):
    print(json.dumps({"error": "timeout"}), file=sys.stderr)
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(18 * 60)

# ---------------------------------------------------------------------------
# OpenAI client from environment variables
# ---------------------------------------------------------------------------
try:
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    hf_token = os.environ.get("HF_TOKEN", "default-token")
    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
    )
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}", file=sys.stderr)
    client = None

MODEL = os.environ.get("MODEL_NAME", "default-model")
if MODEL == "default-model":
    print("Warning: MODEL_NAME not set, using default", file=sys.stderr)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS = 200  # prevents infinite loops from eating all time

SYSTEM_PROMPT = """
You are an operations analyst. You will be given an inbox, ticket queue,
and financial records. Respond ONLY with a valid JSON action object.

For Task 1 (email triage), respond with:
{"action_type": "label_email", "email_id": "...", "label": "billing|onboarding|outage|spam|general", "urgency": 1-5, "next_action": "reply|escalate|archive|forward"}

For Task 2 (ticket routing), respond with:
{"action_type": "route_ticket", "ticket_id": "...", "team": "billing|infra|product|account_management", "escalate": true|false, "draft_message": "..."}

For Task 3 (reconciliation), first use query_db to investigate, then flag discrepancies, then submit report:
{"action_type": "query_db", "sql": "SELECT ..."}
{"action_type": "flag_discrepancy", "invoice_id": "...", "po_id": "...", "discrepancy_type": "...", "explanation": "..."}
{"action_type": "submit_report", "report": {"discrepancies": [...], "summary": "..."}}

Always process urgent items first. Check SLA breach times carefully.
Cross-reference invoice IDs mentioned in ticket descriptions with Task 3 data.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(tag: str, data: dict) -> None:
    """Print a structured log line to stdout."""
    print(f"[{tag}] {json.dumps(data, separators=(',', ':'))}", flush=True)


def _debug(*args, **kwargs) -> None:
    """Print a debug message to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def build_context(obs: Any) -> str:
    """Serialize current observation to a readable prompt string."""
    lines: list[str] = []
    lines.append(f"=== Current Task: {obs.current_task_id} (step {obs.step_count}) ===\n")

    if obs.current_task_id == "task1":
        lines.append("## Inbox (process each email)\n")
        for email in obs.inbox:
            d = email.model_dump(exclude={
                "ground_truth_label",
                "ground_truth_urgency",
                "ground_truth_next_action",
            })
            d["timestamp"] = d["timestamp"].isoformat() if hasattr(d["timestamp"], "isoformat") else str(d["timestamp"])
            lines.append(json.dumps(d, indent=2))
            lines.append("")

    elif obs.current_task_id == "task2":
        lines.append("## Ticket Queue (route each ticket)\n")
        for ticket in obs.tickets:
            d = ticket.model_dump(exclude={
                "ground_truth_team",
                "ground_truth_escalate",
            })
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
    """Parse JSON string from LLM into typed Action model."""
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
        _debug("Failed to decode JSON from model block.", raw)
        # Random safe fallback to prevent crash from hallucinated formats
        return Action(action_type="label_email", email_id="unknown", label="general", urgency=1, next_action="archive")
    except Exception as e:
        _debug(f"Action parsing error: {e}", raw)
        return Action(action_type="label_email", email_id="unknown", label="general", urgency=1, next_action="archive")

def main():
    _debug("Initializing InboxOps environment...")
    
    if client is None:
        _debug("OpenAI client not initialized, exiting.")
        sys.exit(1)

    try:
        env = InboxOpsEnv()
        obs = env.reset(seed=int(os.getenv("SEED", 42)))
    except Exception as e:
        _debug(f"Failed to initialize environment: {e}")
        sys.exit(1)
        
    _debug(f"Using model: {MODEL}")

    start_time = time.time()
    total_reward = 0.0
    step_count = 0
    task_scores: dict[str, float] = {}

    # [START] log
    _log("START", {"episode": 1, "seed": os.getenv("SEED", 42)})

    while step_count < MAX_STEPS:
        context = build_context(obs)

        step_start = time.time()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            _debug(f"  [Step {step_count}] API error: {exc}")
            break

        # Per-step timeout warning
        if time.time() - step_start > 30:
            print(f"[WARN] Slow step {step_count}", file=sys.stderr)

        action = parse_action(raw)
        if action is None:
            _debug(f"  [Step {step_count}] Could not parse response, skipping")
            _debug(f"  Raw response: {raw[:200]}")
            step_count += 1
            continue

        obs, reward, done, info = env.step(action)
        total_reward += reward.value

        # [STEP] log
        _log("STEP", {
            "step": step_count + 1,
            "task": obs.current_task_id,
            "action_type": action.action_type,
            "reward": round(reward.value, 4),
            "total_reward": round(total_reward, 4),
            "done": done,
        })

        if done:
            task_scores = info.get("final_scores", {})
            break

        step_count += 1

    # Warn if step cap reached
    if step_count >= MAX_STEPS:
        print(f"[WARN] Step cap reached", file=sys.stderr)

    # [END] log
    summary = {
        "task1_score": round(task_scores.get("task1", 0.0), 4),
        "task2_score": round(task_scores.get("task2", 0.0), 4),
        "task3_score": round(task_scores.get("task3", 0.0), 4),
        "total_reward": round(total_reward, 4),
        "steps": step_count + 1,
    }
    _log("END", summary)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
