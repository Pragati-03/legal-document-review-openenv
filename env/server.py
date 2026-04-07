from fastapi import FastAPI

app = FastAPI()

class LegalReviewEnv:
    def __init__(self):
        self.state = {"message": "Environment ready"}

    def reset(self):
        return self.state

    def step(self, action):
        return {"action_received": action}, 0, True, {}

env = LegalReviewEnv()   # ✅ now correct

@app.get("/")
def home():
    return {"message": "Legal Review API running 🚀"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)