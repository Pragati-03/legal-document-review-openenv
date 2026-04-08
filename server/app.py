from __future__ import annotations
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.get("/")
def root():
    return {
        "service": "Legal Document Review — OpenEnv",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "tasks": "GET /tasks",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "docs": "GET /docs",
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "legal-review-openenv"}

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

if __name__ == "__main__":
    main()