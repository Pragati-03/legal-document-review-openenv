"""
FastAPI server — exposes the LegalReviewEnv via HTTP following OpenEnv conventions.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env.environment import LegalReviewEnv
from env.models import Action, Observation, Reward

app = FastAPI(
    title="Legal Document Review — OpenEnv",
    description="AI agent environment for reviewing legal contracts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[LegalReviewEnv] = None


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: int = 42


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "legal-review-openenv"}


@app.get("/metadata")
def metadata():
    return {
        "name": "Legal Document Review",
        "description": "AI agent environment for reviewing legal contracts.",
        "version": "1.0.0",
        "task_type": "document_review",
        "domain": "legal",
        "modality": "text",
        "multi_step": True,
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                "recommendation": {"type": "string"},
            },
            "required": ["clause_id", "risk_level", "recommendation"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "clauses": {"type": "array"},
                "document_title": {"type": "string"},
                "step": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "done": {"type": "boolean"},
                "total_reward": {"type": "number"},
            },
        },
    }


@app.post("/mcp")
def mcp(request: Dict[str, Any]):
    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 1),
        "result": {
            "name": "Legal Document Review — OpenEnv",
            "version": "1.0.0",
            "capabilities": ["reset", "step", "state", "schema", "metadata"],
        },
    }


@app.get("/tasks")
def list_tasks():
    from tasks.task_definitions import ALL_TASKS
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "difficulty": t.difficulty,
                "document_title": t.document_title,
                "description": t.description,
                "num_clauses": len(t.clauses),
                "max_steps": t.max_steps,
            }
            for t in ALL_TASKS.values()
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = LegalReviewEnv(task_id=req.task_id, seed=req.seed)
    return _env.reset()


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    obs, reward, done, info = _env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env.state()


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)