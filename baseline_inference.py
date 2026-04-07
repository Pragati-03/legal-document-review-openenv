"""
Baseline inference script for the Legal Document Review OpenEnv environment.
Uses the OpenAI API (gpt-4o) to run an agent on all three tasks and report scores.

Usage:
    export OPENAI_API_KEY="sk-..."
    python baseline_inference.py [--task_id TASK_ID] [--model gpt-4o]

Reproducible baseline scores are printed at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Graceful import — openai is optional when running tests without the key
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from env.environment import LegalReviewEnv
from env.models import (
    Action,
    ActionType,
    IssueCategory,
    IssueSeverity,
)
from tasks.task_definitions import ALL_TASKS


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert legal contract reviewer with 20+ years of experience.
Your task is to review a legal contract clause by clause and identify:
- Ambiguous language that could lead to disputes
- Unfair or one-sided terms
- Missing clauses that should be present
- Compliance risks (GDPR, employment law, etc.)
- Liability exposure
- Intellectual property risks
- Jurisdiction issues
- Inconsistencies across clauses

For each clause, respond with a JSON object with the following schema:
{
  "action_type": one of ["identify_issue", "approve_clause", "suggest_revision", "request_clarification", "submit_review"],
  "clause_id": "the clause ID you are acting on",
  "issue_category": one of ["ambiguous_language", "missing_clause", "unfair_term", "compliance_risk", "inconsistency", "jurisdiction_issue", "liability_exposure", "ip_risk"],
  "issue_severity": one of ["low", "medium", "high", "critical"],
  "issue_description": "clear description of the problem",
  "suggested_revision": "concrete revised text or guidance",
  "reasoning": "your step-by-step reasoning"
}

If a clause is acceptable, use action_type "approve_clause".
When you have reviewed all clauses, use action_type "submit_review".
Always output valid JSON only — no prose outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def build_user_message(obs_dict: Dict[str, Any]) -> str:
    """Construct the user message from the current observation."""
    current = obs_dict.get("current_clause")
    if current is None:
        return json.dumps({
            "instruction": "All clauses reviewed. Please submit your review.",
            "action_hint": "submit_review",
            "issues_found_count": len(obs_dict.get("issues_found", [])),
        })

    return json.dumps({
        "document_title": obs_dict["document_title"],
        "task_id": obs_dict["task_id"],
        "current_clause": current,
        "remaining_clauses": obs_dict["remaining_clauses"],
        "step": obs_dict["step_count"],
        "max_steps": obs_dict["max_steps"],
        "issues_found_so_far": len(obs_dict.get("issues_found", [])),
        "hints": obs_dict.get("hints", []),
        "instruction": (
            "Review the current_clause above. Output a single JSON action object."
        ),
    })


def parse_action_from_response(text: str) -> Optional[Action]:
    """Parse the LLM's JSON response into an Action model."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    try:
        # Normalise action_type
        action_type_str = data.get("action_type", "approve_clause")
        action_type = ActionType(action_type_str)

        # Normalise optional enums
        cat = data.get("issue_category")
        sev = data.get("issue_severity")

        return Action(
            action_type=action_type,
            clause_id=data.get("clause_id"),
            issue_category=IssueCategory(cat) if cat else None,
            issue_severity=IssueSeverity(sev) if sev else None,
            issue_description=data.get("issue_description"),
            suggested_revision=data.get("suggested_revision"),
            clarification_question=data.get("clarification_question"),
            reasoning=data.get("reasoning"),
        )
    except (ValueError, KeyError) as e:
        print(f"  [warn] Could not parse action: {e} — defaulting to approve")
        return None


def run_agent_on_task(
    client: "OpenAI",
    task_id: str,
    model: str = "gpt-4o",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the agent on a single task and return results."""
    env = LegalReviewEnv(task_id=task_id, seed=42)
    obs = env.reset()

    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    final_grade = None
    total_step_rewards = []
    steps_taken = 0
    done = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Document: {obs.document_title}")
        print(f"Total clauses: {obs.progress.total_clauses}")
        print(f"{'='*60}")

    while not done and steps_taken < env._task.max_steps:
        obs_dict = obs.model_dump()

        # Build user message
        user_msg = build_user_message(obs_dict)
        conversation.append({"role": "user", "content": user_msg})

        # Call the model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                temperature=0.0,   # deterministic
                max_tokens=800,
            )
            assistant_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [error] OpenAI API call failed: {e}")
            break

        conversation.append({"role": "assistant", "content": assistant_text})

        # Parse action
        action = parse_action_from_response(assistant_text)
        if action is None:
            # Default: approve current clause
            current = obs.current_clause
            if current:
                action = Action(
                    action_type=ActionType.APPROVE_CLAUSE,
                    clause_id=current.clause_id,
                )
            else:
                action = Action(action_type=ActionType.SUBMIT_REVIEW)

        # Step
        obs, reward, done, info = env.step(action)
        steps_taken += 1
        total_step_rewards.append(reward.step_reward)

        if verbose:
            print(
                f"  Step {steps_taken:02d} | {action.action_type.value:25s} | "
                f"clause={action.clause_id or 'n/a':6s} | "
                f"step_rwd={reward.step_reward:+.3f} | "
                f"cumulative={reward.cumulative_reward:+.3f}"
            )

        if "final_grade" in info:
            final_grade = info["final_grade"]

        # Small delay to avoid rate limiting
        time.sleep(0.2)

    # If loop ended without explicit submission, get final grade
    if final_grade is None:
        from graders.grader import grade_episode
        grade = grade_episode(
            task=env._task,
            identified_issues=env._identified_issues,
            approved_clauses=env._approved_clauses,
            skipped_clauses=env._skipped_clauses,
            step_count=steps_taken,
            submitted=env._submitted,
        )
        final_grade = grade.__dict__

    return {
        "task_id": task_id,
        "difficulty": ALL_TASKS[task_id].difficulty,
        "steps_taken": steps_taken,
        "final_score": final_grade.get("score", 0.0),
        "breakdown": final_grade.get("breakdown", {}),
        "explanation": final_grade.get("explanation", ""),
        "total_step_reward": round(sum(total_step_rewards), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Legal Review OpenEnv Baseline")
    parser.add_argument("--task_id", default=None, help="Run a specific task only")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    if not OPENAI_AVAILABLE:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    task_ids = [args.task_id] if args.task_id else list(ALL_TASKS.keys())

    results = []
    for tid in task_ids:
        result = run_agent_on_task(client, tid, model=args.model, verbose=args.verbose)
        results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("BASELINE RESULTS")
    print(f"{'='*70}")
    print(f"{'Task ID':<30} {'Difficulty':<10} {'Steps':<8} {'Score':<8} {'Explanation'}")
    print(f"{'-'*70}")
    for r in results:
        print(
            f"{r['task_id']:<30} {r['difficulty']:<10} {r['steps_taken']:<8} "
            f"{r['final_score']:.4f}   {r['explanation'][:50]}..."
        )
    print(f"{'='*70}")

    # Machine-readable output
    print("\nJSON Results:")
    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
