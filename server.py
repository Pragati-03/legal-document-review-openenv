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

# Hardcode /app as first path entry
sys.path = ["/app"] + [p for p in sys.path if p not in ("", ".", "/app")]

# Import everything at startup so errors are visible in build logs
from env.models import Action, Observation, Reward
from env.environment import LegalReviewEnv
from tasks.task_definitions import ALL_TASKS

app = FastAPI(title="Legal Document Review — OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_static_dir = "/app/static"
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": traceback.format_exc()})

_env: Optional[LegalReviewEnv] = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: int = 42

@app.get("/")
def root():
    idx = "/app/static/index.html"
    if os.path.isfile(idx):
        return FileResponse(idx)
    return {"status": "ok", "message": "Legal Document Review OpenEnv API"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "legal-review-openenv"}

@app.get("/debug")
def debug():
    return {
        "cwd": os.getcwd(),
        "sys_path": sys.path[:6],
        "root_files": os.listdir("/app"),
        "tasks_files": os.listdir("/app/tasks") if os.path.isdir("/app/tasks") else "MISSING",
        "env_files": os.listdir("/app/env") if os.path.isdir("/app/env") else "MISSING",
        "graders_files": os.listdir("/app/graders") if os.path.isdir("/app/graders") else "MISSING",
        "all_tasks_loaded": list(ALL_TASKS.keys()),
        "sample_task_fields": {
            k: getattr(list(ALL_TASKS.values())[0], k, "MISSING")
            for k in ["task_id", "difficulty", "document_title", "max_steps"]
        } if ALL_TASKS else {},
    }

@app.get("/tasks")
def list_tasks():
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

@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    try:
        _env = LegalReviewEnv(task_id=req.task_id, seed=req.seed)
        obs = _env.reset()
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}\n{traceback.format_exc()}")

@app.post("/step")
def step(request: Dict[str, Any]):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        action = Action(**request)
        obs, reward, done, info = _env.step(action)
        return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}\n{traceback.format_exc()}")

@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        return _env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)