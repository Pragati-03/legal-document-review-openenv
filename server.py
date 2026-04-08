"""
FastAPI server — Legal Document Review OpenEnv
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── CRITICAL: hardcode /app as first on path ────────────────────────────────
if "/app" not in sys.path:
    sys.path.insert(0, "/app")
# Remove any path entries that are just "." or "" which cause root imports
sys.path = [p for p in sys.path if p not in ("", ".")]
sys.path.insert(0, "/app")

app = FastAPI(title="Legal Document Review — OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_static_dir = "/app/static"
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": traceback.format_exc()})

_env = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: int = 42


def _load_tasks_fresh():
    """Load tasks/task_definitions.py directly by file path — bypasses sys.modules cache."""
    spec = importlib.util.spec_from_file_location(
        "tasks.task_definitions",
        "/app/tasks/task_definitions.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    info = {}
    info["cwd"] = os.getcwd()
    info["sys_path"] = sys.path[:8]
    info["root_files"] = os.listdir("/app")
    info["tasks_files"] = os.listdir("/app/tasks") if os.path.isdir("/app/tasks") else "MISSING"
    info["env_files"] = os.listdir("/app/env") if os.path.isdir("/app/env") else "MISSING"
    info["graders_files"] = os.listdir("/app/graders") if os.path.isdir("/app/graders") else "MISSING"

    # Test direct file load
    try:
        td = _load_tasks_fresh()
        info["task_definitions_loaded_from"] = "/app/tasks/task_definitions.py (direct)"
        info["TaskDefinition_fields"] = list(td.TaskDefinition.__dataclass_fields__.keys())
        info["num_tasks"] = len(td.ALL_TASKS)
        info["task_ids"] = list(td.ALL_TASKS.keys())
    except Exception as e:
        info["task_definitions_error"] = str(e)

    return info

@app.get("/tasks")
def list_tasks():
    try:
        td = _load_tasks_fresh()
        ALL_TASKS = td.ALL_TASKS
        return {
            "tasks": [
                {
                    "task_id": getattr(t, "task_id", "?"),
                    "difficulty": getattr(t, "difficulty", "?"),
                    "document_title": getattr(t, "document_title", "?"),
                    "description": getattr(t, "description", ""),
                    "num_clauses": len(getattr(t, "clauses", [])),
                    "max_steps": getattr(t, "max_steps", 30),
                }
                for t in ALL_TASKS.values()
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed: {str(e)}\n{traceback.format_exc()}"
        )

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
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        from env.models import Action
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