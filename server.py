"""
FastAPI server — Legal Document Review OpenEnv
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# make sure /app is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# static UI
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": tb},
    )

# session store
_env = None

# pydantic models
class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: int = 42

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]

# endpoints

@app.get("/")
def root():
    idx = os.path.join(_static_dir, "index.html")
    if os.path.isfile(idx):
        return FileResponse(idx)
    return {"status": "ok", "message": "Legal Document Review OpenEnv API"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "legal-review-openenv"}

@app.get("/debug")
def debug():
    """Import diagnostics — shows exactly what is failing."""
    results = {}
    for mod in ["env.models", "env.environment", "tasks.task_definitions", "graders.grader"]:
        try:
            __import__(mod)
            results[mod] = "OK"
        except Exception as e:
            results[mod] = f"FAILED: {str(e)}"
    results["cwd"] = os.getcwd()
    results["files"] = os.listdir(".")
    results["env_files"] = os.listdir("env") if os.path.isdir("env") else "missing"
    results["tasks_files"] = os.listdir("tasks") if os.path.isdir("tasks") else "missing"
    results["graders_files"] = os.listdir("graders") if os.path.isdir("graders") else "missing"
    return results

@app.get("/tasks")
def list_tasks():
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tasks: {str(e)}\n{traceback.format_exc()}")

@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    try:
        from env.environment import LegalReviewEnv
        _env = LegalReviewEnv(task_id=req.task_id, seed=req.seed)
        obs = _env.reset()
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}\n{traceback.format_exc()}")

@app.post("/step")
def step(request: Dict[str, Any]):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    try:
        from env.models import Action
        action = Action(**request)
        obs, reward, done, info = _env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward.model_dump(),
            done=done,
            info=info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}\n{traceback.format_exc()}")

@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    try:
        return _env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)