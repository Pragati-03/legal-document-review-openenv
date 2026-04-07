from fastapi import FastAPI

app = FastAPI()

env = LegalReviewEnv()

@app.get("/")
def home():
    return {"message": "Legal Review API running 🚀"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)

class LegalReviewEnv:
    def __init__(self):
        self.state = {"message": "Environment ready"}

    def reset(self):
        return self.state

    def step(self, action):
        # Just echo the action for now
        return {"action_received": action}, 0, True, {}