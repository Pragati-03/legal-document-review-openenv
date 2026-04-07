#!/usr/bin/env python3
"""
openenv_validate.py — validates this environment against the OpenEnv spec.
Simulates what `openenv validate` checks.

Run:
    python openenv_validate.py

Exit code 0 = all checks pass.
Exit code 1 = one or more checks failed.
"""
import sys
import json
import yaml
from pathlib import Path

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []
warnings = []

def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f": {detail}" if detail else ""))
        errors.append(name)

def warn(name: str, detail: str = "") -> None:
    print(f"  {WARN} {name}" + (f": {detail}" if detail else ""))
    warnings.append(name)


print("=" * 60)
print("OpenEnv Validator — Legal Document Review")
print("=" * 60)

# ------------------------------------------------------------------
print("\n[1] YAML Metadata")
# ------------------------------------------------------------------
yaml_path = Path("openenv.yaml")
check("openenv.yaml exists", yaml_path.exists())
if yaml_path.exists():
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    check("name field present", "name" in meta, str(meta.get("name")))
    check("version field present", "version" in meta)
    check("description field present", "description" in meta)
    check("api.reset defined", "api" in meta and "reset" in meta.get("api", {}))
    check("api.step defined", "api" in meta and "step" in meta.get("api", {}))
    check("api.state defined", "api" in meta and "state" in meta.get("api", {}))
    check("observation_space defined", "observation_space" in meta)
    check("action_space defined", "action_space" in meta)
    check("reward section defined", "reward" in meta)
    check("tasks list defined", "tasks" in meta and len(meta.get("tasks", [])) >= 3,
          f"found {len(meta.get('tasks', []))} tasks")
    check("tags include openenv", "openenv" in meta.get("tags", []))

# ------------------------------------------------------------------
print("\n[2] Pydantic Models")
# ------------------------------------------------------------------
try:
    from env.models import Observation, Action, Reward, ActionType, IssueCategory, IssueSeverity
    check("env.models imports cleanly", True)
    check("Observation is Pydantic BaseModel", hasattr(Observation, "model_fields"))
    check("Action is Pydantic BaseModel", hasattr(Action, "model_fields"))
    check("Reward is Pydantic BaseModel", hasattr(Reward, "model_fields"))

    # Field checks
    obs_fields = set(Observation.model_fields.keys())
    check("Observation.task_id", "task_id" in obs_fields)
    check("Observation.current_clause", "current_clause" in obs_fields)
    check("Observation.step_count", "step_count" in obs_fields)
    check("Observation.max_steps", "max_steps" in obs_fields)
    check("Observation.progress", "progress" in obs_fields)

    action_fields = set(Action.model_fields.keys())
    check("Action.action_type", "action_type" in action_fields)
    check("Action.clause_id", "clause_id" in action_fields)

    reward_fields = set(Reward.model_fields.keys())
    check("Reward.step_reward", "step_reward" in reward_fields)
    check("Reward.cumulative_reward", "cumulative_reward" in reward_fields)
    check("Reward.feedback", "feedback" in reward_fields)

    # Enum checks
    check("ActionType enum has identify_issue", "identify_issue" in [e.value for e in ActionType])
    check("ActionType enum has submit_review", "submit_review" in [e.value for e in ActionType])
    check("IssueCategory enum defined", len(list(IssueCategory)) >= 4)
    check("IssueSeverity enum defined", len(list(IssueSeverity)) == 4)

except ImportError as e:
    check("env.models imports cleanly", False, str(e))

# ------------------------------------------------------------------
print("\n[3] Environment Class")
# ------------------------------------------------------------------
try:
    from env.environment import LegalReviewEnv
    check("LegalReviewEnv imports cleanly", True)
    check("reset() method exists", hasattr(LegalReviewEnv, "reset"))
    check("step() method exists", hasattr(LegalReviewEnv, "step"))
    check("state() method exists", hasattr(LegalReviewEnv, "state"))
except ImportError as e:
    check("LegalReviewEnv imports cleanly", False, str(e))

# ------------------------------------------------------------------
print("\n[4] reset() returns Observation")
# ------------------------------------------------------------------
try:
    from env.environment import LegalReviewEnv
    from env.models import Observation
    env = LegalReviewEnv("task_easy_freelance")
    obs = env.reset()
    check("reset() returns Observation instance", isinstance(obs, Observation))
    check("Observation is valid (task_id present)", bool(obs.task_id))
    check("Observation has progress", obs.progress is not None)
    check("Observation has current_clause on reset", obs.current_clause is not None)
except Exception as e:
    check("reset() returns Observation instance", False, str(e))

# ------------------------------------------------------------------
print("\n[5] step() returns (Observation, Reward, bool, dict)")
# ------------------------------------------------------------------
try:
    from env.models import Action, ActionType, IssueCategory, IssueSeverity, Reward
    env = LegalReviewEnv("task_easy_freelance")
    env.reset()
    action = Action(
        action_type=ActionType.APPROVE_CLAUSE,
        clause_id="E01",
    )
    result = env.step(action)
    check("step() returns 4-tuple", isinstance(result, tuple) and len(result) == 4)
    obs2, rwd, done, info = result
    check("step() obs is Observation", isinstance(obs2, Observation))
    check("step() reward is Reward", isinstance(rwd, Reward))
    check("step() done is bool", isinstance(done, bool))
    check("step() info is dict", isinstance(info, dict))
    check("Reward.step_reward in [-1, 1]", -1.0 <= rwd.step_reward <= 1.0,
          str(rwd.step_reward))
    check("Reward.feedback is string", isinstance(rwd.feedback, str))
    check("info has episode_id", "episode_id" in info)
    check("info has task_id", "task_id" in info)
except Exception as e:
    check("step() returns 4-tuple", False, str(e))

# ------------------------------------------------------------------
print("\n[6] state() returns dict")
# ------------------------------------------------------------------
try:
    env3 = LegalReviewEnv("task_easy_freelance")
    env3.reset()
    s = env3.state()
    check("state() returns dict", isinstance(s, dict))
    check("state has episode_id", "episode_id" in s)
    check("state has task_id", "task_id" in s)
    check("state has step_count", "step_count" in s)
    check("state has done", "done" in s)
    check("state has identified_issues", "identified_issues" in s)
    # Verify state is JSON-serialisable
    json.dumps(s)
    check("state() is JSON-serialisable", True)
except Exception as e:
    check("state() returns dict", False, str(e))

# ------------------------------------------------------------------
print("\n[7] Tasks & Graders")
# ------------------------------------------------------------------
try:
    from tasks.task_definitions import ALL_TASKS
    check("≥3 tasks defined", len(ALL_TASKS) >= 3, f"found {len(ALL_TASKS)}")

    difficulties = {t.difficulty for t in ALL_TASKS.values()}
    check("easy task exists", "easy" in difficulties)
    check("medium task exists", "medium" in difficulties)
    check("hard task exists", "hard" in difficulties)

    from graders.grader import grade_episode
    from env.models import IdentifiedIssue, IssueCategory, IssueSeverity

    for task in ALL_TASKS.values():
        # Grade with empty submission
        result = grade_episode(
            task=task,
            identified_issues=[],
            approved_clauses=[],
            skipped_clauses=[],
            step_count=0,
            submitted=False,
        )
        check(
            f"grader({task.task_id}) score in [0,1]",
            0.0 <= result.score <= 1.0,
            str(result.score),
        )
        check(
            f"grader({task.task_id}) has breakdown",
            isinstance(result.breakdown, dict) and len(result.breakdown) > 0,
        )
        # Verify determinism (call twice)
        result2 = grade_episode(
            task=task, identified_issues=[], approved_clauses=[],
            skipped_clauses=[], step_count=0, submitted=False,
        )
        check(
            f"grader({task.task_id}) is deterministic",
            result.score == result2.score,
        )

except Exception as e:
    check("Tasks & Graders", False, str(e))

# ------------------------------------------------------------------
print("\n[8] Episode lifecycle")
# ------------------------------------------------------------------
try:
    from env.models import Action, ActionType
    env4 = LegalReviewEnv("task_easy_freelance")
    obs4 = env4.reset()
    check("reset() resets step_count to 0", obs4.step_count == 0)

    # Force termination via submit
    env4.step(Action(action_type=ActionType.SUBMIT_REVIEW))
    state4 = env4.state()
    check("done=True after submit_review", state4["done"] is True)

    # Verify RuntimeError on step after done
    raised = False
    try:
        env4.step(Action(action_type=ActionType.SUBMIT_REVIEW))
    except RuntimeError:
        raised = True
    check("RuntimeError raised when stepping after done", raised)

    # Verify reset restores fresh state
    obs5 = env4.reset()
    check("reset() restores done=False", env4.state()["done"] is False)
    check("reset() restores step_count=0", obs5.step_count == 0)
except Exception as e:
    check("Episode lifecycle", False, str(e))

# ------------------------------------------------------------------
print("\n[9] Partial reward signals")
# ------------------------------------------------------------------
try:
    from env.models import Action, ActionType, IssueCategory, IssueSeverity
    env5 = LegalReviewEnv("task_easy_freelance")
    env5.reset()

    # Good action → positive reward
    good_action = Action(
        action_type=ActionType.IDENTIFY_ISSUE,
        clause_id="E02",
        issue_category=IssueCategory.AMBIGUOUS_LANGUAGE,
        issue_severity=IssueSeverity.HIGH,
        issue_description="Deliverables and scope are undefined",
        suggested_revision="Specify exact deliverables and milestones",
    )
    _, rwd_good, _, _ = env5.step(good_action)
    check("Good action → positive step reward", rwd_good.step_reward > 0,
          str(rwd_good.step_reward))

    env6 = LegalReviewEnv("task_easy_freelance")
    env6.reset()
    bad_action = Action(
        action_type=ActionType.APPROVE_CLAUSE,
        clause_id="E02",  # This is actually a GT issue — should be flagged
    )
    _, rwd_bad, _, _ = env6.step(bad_action)
    check("Bad action → negative step reward", rwd_bad.step_reward < 0,
          str(rwd_bad.step_reward))

    env7 = LegalReviewEnv("task_easy_freelance")
    env7.reset()
    safe_action = Action(action_type=ActionType.APPROVE_CLAUSE, clause_id="E01")
    _, rwd_safe, _, _ = env7.step(safe_action)
    check("Safe approve → positive step reward", rwd_safe.step_reward > 0,
          str(rwd_safe.step_reward))
except Exception as e:
    check("Partial reward signals", False, str(e))

# ------------------------------------------------------------------
print("\n[10] Dockerfile & deployment files")
# ------------------------------------------------------------------
check("Dockerfile exists", Path("Dockerfile").exists())
check("requirements.txt exists", Path("requirements.txt").exists())
check("server.py exists", Path("server.py").exists())
check("baseline_inference.py exists", Path("baseline_inference.py").exists())

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 60)
if errors:
    print(f"{FAIL} VALIDATION FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"   - {e}")
    if warnings:
        print(f"{WARN} {len(warnings)} warning(s): {warnings}")
    sys.exit(1)
else:
    print(f"{PASS} ALL CHECKS PASSED ({len(ALL_TASKS)} tasks, 0 errors)")
    if warnings:
        print(f"{WARN} {len(warnings)} warning(s): {warnings}")
    print("=" * 60)
    sys.exit(0)
