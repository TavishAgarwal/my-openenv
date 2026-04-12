"""FastAPI server for InboxOps HF Space deployment.

Exposes /health, /reset, /step, /state, and /session/{session_id} endpoints
so judges can ping the Space and run episodes over HTTP.

Each caller gets an isolated session via session_id, preventing concurrent
requests from corrupting each other's environment state.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException

from environment.env import InboxOpsEnv
from environment.models import (
    FlagDiscrepancyAction,
    LabelEmailAction,
    QueryDatabaseAction,
    RouteTicketAction,
    SubmitReportAction,
)

app = FastAPI()

# Session store: each session_id maps to an isolated InboxOpsEnv instance.
# For production use, add TTL-based eviction (see /session/{session_id} DELETE).
_sessions: dict[str, InboxOpsEnv] = {}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "name": "inboxops",
        "version": "1.0.0",
        "active_sessions": len(_sessions),
    }


@app.post("/reset")
def reset(body: dict = {}):
    """Create or re-initialise a session.

    Request body (all optional):
        session_id: str — reuse an existing session ID, or omit for a new one.
        seed: int — environment seed (default 42).

    Returns the session_id and initial observation.
    """
    session_id = body.get("session_id") or str(uuid.uuid4())
    seed = body.get("seed", 42)
    env = InboxOpsEnv(seed=seed)
    obs = env.reset()
    _sessions[session_id] = env
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(body: dict):
    """Take a single action in an existing session.

    Request body (required):
        session_id: str
        action_type: str — one of label_email, route_ticket, query_db,
                           flag_discrepancy, submit_report
        ... (action-specific fields)
    """
    session_id = body.get("session_id")
    if not session_id or session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing session_id. Call /reset first.",
        )
    env = _sessions[session_id]

    action_map = {
        "label_email": LabelEmailAction,
        "route_ticket": RouteTicketAction,
        "query_db": QueryDatabaseAction,
        "flag_discrepancy": FlagDiscrepancyAction,
        "submit_report": SubmitReportAction,
    }
    action_type = body.get("action_type")
    ActionClass = action_map.get(action_type)
    if not ActionClass:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action_type: {action_type}",
        )

    typed_action = ActionClass(**body)
    obs, reward, done, info = env.step(typed_action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str):
    """Return the current observation for a session."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="Unknown session_id. Call /reset first.",
        )
    return _sessions[session_id].state().model_dump()


@app.delete("/session/{session_id}")
def close_session(session_id: str):
    """Close a session and free its resources."""
    _sessions.pop(session_id, None)
    return {"status": "closed", "session_id": session_id}
