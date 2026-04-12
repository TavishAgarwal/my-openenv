"""Baseline inference script for InboxOps.

Reads HF_TOKEN from environment (used as OpenAI API key against
Hugging Face Inference API). Runs one full episode through all 3 tasks
and prints a score table.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI

# Ensure project root is on the path when run from Docker or directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import InboxOpsEnv
from environment.models import (
    Action,
    FlagDiscrepancyAction,
    LabelEmailAction,
    QueryDatabaseAction,
    RouteTicketAction,
    SubmitReportAction,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=HF_TOKEN,
)
MODEL = "Qwen/Qwen2.5-72B-Instruct"

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


def build_context(obs: Any) -> str:
    """Serialize current observation to a readable prompt string.

    Excludes ground-truth fields so the agent only sees what it should.
    """
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
    """Parse JSON string from LLM into typed Action model.

    Returns None if the response cannot be parsed, allowing the caller
    to skip the step gracefully.
    """
    try:
        # Strip markdown code fences if present
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
            print(f"  Unknown action_type: {action_type}")
            return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"  Failed to parse action: {exc}")
        return None


def run_episode(seed: int = 42) -> dict:
    """Run a complete episode using the HF inference API."""
    env = InboxOpsEnv(seed=seed)
    obs = env.reset()
    total_reward = 0.0
    task_scores: dict[str, float] = {}
    max_steps = 200  # safety limit

    step = 0
    while step < max_steps:
        context = build_context(obs)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [Step {step}] API error: {exc}")
            break

        try:
            action = parse_action(raw)
        except Exception as exc:
            print(f"  [Step {step}] Parse error: {exc}")
            print(f"  Raw response: {raw[:200]}")
            step += 1
            continue

        if action is None:
            print(f"  [Step {step}] Could not parse response, skipping")
            print(f"  Raw response: {raw[:200]}")
            step += 1
            continue

        obs, reward, done, info = env.step(action)
        total_reward += reward.value

        print(f"  [Step {step}] {action.action_type} → reward={reward.value:.3f}")

        if done:
            task_scores = info.get("final_scores", {})
            break

        step += 1

    print("\n" + "=" * 40)
    print("  InboxOps Baseline Results")
    print("=" * 40)
    for task, score in task_scores.items():
        print(f"  {task}: {score:.3f}")
    print(f"  Total: {total_reward:.3f}")
    print("=" * 40)

    return task_scores


if __name__ == "__main__":
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Set it to use HuggingFace Inference API.")
        print("Example: export HF_TOKEN=hf_xxxxxxxxx")
        print("Running in dry-run mode (will fail on API calls).\n")

    run_episode()
