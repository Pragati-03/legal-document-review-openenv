from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import uvicorn

class LegalReviewEnv:
    def __init__(self):
        self.state = {"message": "Environment ready"}
    def reset(self):
        return self.state
    def step(self, action):
        return {"action_received": action}, 0, True, {}

env = LegalReviewEnv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm env before accepting traffic
    env.reset()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home(request: Request):
    return {"message": "Legal Review API running 🚀"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)