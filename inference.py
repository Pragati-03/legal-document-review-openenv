"""
inference.py — Required entry point for OpenEnv evaluation.
Runs a baseline agent against the LegalReviewEnv.
"""
import requests
import os

BASE_URL = os.environ.get("SPACE_URL", "http://0.0.0.0:7860")


def run_inference(task_id: str = None, seed: int = 42) -> dict:
    # Reset environment
    reset_resp = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        headers={"Content-Type": "application/json"},
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        # Baseline action: flag the first clause as risky
        clauses = obs.get("clauses", [])
        if not clauses:
            break

        action = {
            "clause_id": clauses[0].get("clause_id", "clause_0"),
            "risk_level": "high",
            "recommendation": "Review this clause carefully.",
        }

        step_resp = requests.post(
            f"{BASE_URL}/step",
            json=action,
            headers={"Content-Type": "application/json"},
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result["observation"]
        total_reward += result["reward"].get("score", 0)
        done = result["done"]
        steps += 1

    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "steps": steps,
        "done": done,
    }


if __name__ == "__main__":
    result = run_inference()
    print(result)