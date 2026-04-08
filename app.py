"""FastAPI server for InboxOps HF Space deployment.

Exposes /health, /reset, /step, and /state endpoints so judges can
ping the Space and run episodes over HTTP.
"""

from __future__ import annotations

from fastapi import FastAPI

from environment.env import InboxOpsEnv
from environment.models import (
    FlagDiscrepancyAction,
    LabelEmailAction,
    QueryDatabaseAction,
    RouteTicketAction,
    SubmitReportAction,
)

app = FastAPI()
_env = InboxOpsEnv(seed=42)


@app.get("/health")
def health():
    return {"status": "ok", "name": "inboxops", "version": "1.0.0"}


@app.post("/reset")
def reset():
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: dict):
    action_map = {
        "label_email": LabelEmailAction,
        "route_ticket": RouteTicketAction,
        "query_db": QueryDatabaseAction,
        "flag_discrepancy": FlagDiscrepancyAction,
        "submit_report": SubmitReportAction,
    }
    action_type = action.get("action_type")
    ActionClass = action_map.get(action_type)
    if not ActionClass:
        return {"error": f"Unknown action_type: {action_type}"}
    typed_action = ActionClass(**action)
    obs, reward, done, info = _env.step(typed_action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return _env.state().model_dump()
