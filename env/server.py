from fastapi import FastAPI, Request

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

# 🔥 CATCH EVERYTHING (important)
@app.get("/{full_path:path}")
def catch_all(full_path: str):
    return {"message": "API running", "path": full_path}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)