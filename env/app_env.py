class LegalReviewEnv:
    def __init__(self):
        self.state = {"message": "Environment ready"}

    def reset(self):
        return self.state

    def step(self, action):
        # Just echo the action for now
        return {"action_received": action}, 0, True, {}