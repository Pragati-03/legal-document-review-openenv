from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class LegalReviewEnv:
    def __init__(self):
        self.state = {"message": "Environment ready"}

    def reset(self):
        return self.state

    def step(self, action):
        return {"action_received": action}, 0, True, {}

env = LegalReviewEnv()

@app.get("/")
def home(request: Request):
    return {"message": "Legal Review API running 🚀"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)

# Catch-all MUST come last, and only for truly unknown paths
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def catch_all(full_path: str, request: Request):
    return JSONResponse(
        status_code=200,
        content={"message": "API running", "path": full_path, "query": str(request.query_params)}
    )