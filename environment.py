"""
Legal Document Review — OpenEnv Environment
============================================
Implements the OpenEnv standard API:
    reset()  → Observation
    step()   → (Observation, Reward, done: bool, info: dict)
    state()  → dict

Domain: AI agent acts as a legal contract reviewer, identifying issues,
suggesting revisions, and approving or flagging clauses in realistic contracts.
"""
from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action,
    ActionType,
    Clause,
    IdentifiedIssue,
    IssueCategory,
    IssueSeverity,
    Observation,
    Reward,
    ReviewProgress,
)
from graders.grader import grade_episode, step_reward
from tasks.task_definitions import ALL_TASKS, TaskDefinition


class LegalReviewEnv:
    """
    OpenEnv-compatible Legal Document Review environment.

    Supports three tasks (easy / medium / hard) selectable at construction time.
    Implements the full OpenEnv spec: reset(), step(), state().
    """

    # Registry of valid task IDs
    TASK_IDS = list(ALL_TASKS.keys())

    def __init__(self, task_id: Optional[str] = None, seed: int = 42):
        """
        Parameters
        ----------
        task_id : str or None
            One of the registered task IDs. If None, uses the easy task.
        seed : int
            Random seed (environment is deterministic so this affects only
            shuffle order if randomisation is added later).
        """
        if task_id is None:
            task_id = "task_easy_freelance"
        if task_id not in ALL_TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid options: {list(ALL_TASKS.keys())}"
            )
        self._task_id = task_id
        self._seed = seed
        self._task: TaskDefinition = ALL_TASKS[task_id]
        self._episode_id: str = ""

        # Mutable episode state — initialised by reset()
        self._clauses: List[Clause] = []
        self._clause_index: int = 0
        self._identified_issues: List[IdentifiedIssue] = []
        self._approved_clauses: List[str] = []
        self._skipped_clauses: List[str] = []
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._submitted: bool = False
        self._clarifications: List[str] = []
        self._history: List[Dict[str, Any]] = []

        # Initialise with a reset so state() is valid before first reset() call
        self.reset()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._episode_id = str(uuid.uuid4())
        self._clauses = [
            Clause(**c) for c in self._task.clauses
        ]
        self._clause_index = 0
        self._identified_issues = []
        self._approved_clauses = []
        self._skipped_clauses = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._submitted = False
        self._clarifications = []
        self._history = []

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action and return (observation, reward, done, info).

        Parameters
        ----------
        action : Action
            Typed Pydantic action model.

        Returns
        -------
        observation : Observation
        reward      : Reward  (partial signal every step)
        done        : bool
        info        : dict    (diagnostic information)
        """
        if self._done:
            raise RuntimeError(
                "Episode has ended. Call reset() before stepping again."
            )

        self._step_count += 1
        action_type = action.action_type

        # Snapshot BEFORE dispatch (for duplicate detection in step_reward)
        pre_action_issues = list(self._identified_issues)

        # ---- Dispatch action ----
        if action_type == ActionType.SUBMIT_REVIEW:
            self._submitted = True
            self._done = True

        elif action_type == ActionType.IDENTIFY_ISSUE:
            if action.clause_id and action.issue_category and action.issue_severity:
                issue = IdentifiedIssue(
                    clause_id=action.clause_id,
                    category=action.issue_category,
                    severity=action.issue_severity,
                    description=action.issue_description or "",
                    suggested_revision=action.suggested_revision,
                )
                self._identified_issues.append(issue)
                self._advance_clause_if_current(action.clause_id)

        elif action_type == ActionType.SUGGEST_REVISION:
            if action.clause_id and action.suggested_revision:
                # Update existing issue or create lightweight entry
                existing = [
                    i for i in self._identified_issues if i.clause_id == action.clause_id
                ]
                if existing:
                    existing[-1].suggested_revision = action.suggested_revision
                else:
                    issue = IdentifiedIssue(
                        clause_id=action.clause_id,
                        category=action.issue_category or IssueCategory.AMBIGUOUS_LANGUAGE,
                        severity=action.issue_severity or IssueSeverity.MEDIUM,
                        description=action.issue_description or "",
                        suggested_revision=action.suggested_revision,
                    )
                    self._identified_issues.append(issue)
                self._advance_clause_if_current(action.clause_id)

        elif action_type == ActionType.APPROVE_CLAUSE:
            if action.clause_id and action.clause_id not in self._approved_clauses:
                self._approved_clauses.append(action.clause_id)
                self._advance_clause_if_current(action.clause_id)

        elif action_type == ActionType.SKIP_CLAUSE:
            if action.clause_id and action.clause_id not in self._skipped_clauses:
                self._skipped_clauses.append(action.clause_id)
                self._advance_clause_if_current(action.clause_id)

        elif action_type == ActionType.REQUEST_CLARIFICATION:
            if action.clarification_question:
                self._clarifications.append(action.clarification_question)

        # ---- Auto-advance when all clauses reviewed ----
        reviewed = set(
            [i.clause_id for i in self._identified_issues]
            + self._approved_clauses
            + self._skipped_clauses
        )
        all_clause_ids = {c.clause_id for c in self._clauses}
        if reviewed >= all_clause_ids and not self._submitted:
            # Prompt submission but don't force it
            pass

        # ---- Max steps exceeded → force done ----
        if self._step_count >= self._task.max_steps:
            self._done = True

        # ---- Compute step reward ----
        step_rwd, feedback = step_reward(
            task=self._task,
            action_type=action_type.value,
            clause_id=action.clause_id or "",
            issue_category=action.issue_category.value if action.issue_category else None,
            issue_severity=action.issue_severity.value if action.issue_severity else None,
            issue_description=action.issue_description,
            suggested_revision=action.suggested_revision,
            cumulative_issues=pre_action_issues,
            step_count=self._step_count,
        )

        # ---- Final episode bonus on done ----
        episode_bonus = 0.0
        episode_grade = None
        if self._done:
            episode_grade = grade_episode(
                task=self._task,
                identified_issues=self._identified_issues,
                approved_clauses=self._approved_clauses,
                skipped_clauses=self._skipped_clauses,
                step_count=self._step_count,
                submitted=self._submitted,
            )
            # Episode score mapped to [-0.5, 0.5] delta on top of step rewards
            episode_bonus = (episode_grade.score - 0.5) * 0.5
            feedback += f" | EPISODE COMPLETE — final score: {episode_grade.score:.4f}"

        self._cumulative_reward += step_rwd + episode_bonus
        self._cumulative_reward = round(self._cumulative_reward, 4)

        reward = Reward(
            step_reward=round(step_rwd, 4),
            cumulative_reward=self._cumulative_reward,
            reward_breakdown={
                "step_component": round(step_rwd, 4),
                "episode_bonus": round(episode_bonus, 4),
            },
            feedback=feedback,
        )

        # ---- Record history ----
        self._history.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": reward.model_dump(),
            "done": self._done,
        })

        obs = self._make_observation()
        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "step": self._step_count,
            "task_id": self._task_id,
        }
        if episode_grade is not None:
            info["final_grade"] = episode_grade.__dict__

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the current environment state."""
        current = self._current_clause()
        return {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "task_difficulty": self._task.difficulty,
            "document_title": self._task.document_title,
            "step_count": self._step_count,
            "max_steps": self._task.max_steps,
            "done": self._done,
            "submitted": self._submitted,
            "current_clause": current.model_dump() if current else None,
            "clause_index": self._clause_index,
            "total_clauses": len(self._clauses),
            "identified_issues": [i.model_dump() for i in self._identified_issues],
            "approved_clauses": self._approved_clauses,
            "skipped_clauses": self._skipped_clauses,
            "cumulative_reward": self._cumulative_reward,
            "clarifications": self._clarifications,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_clause(self) -> Optional[Clause]:
        if self._clause_index < len(self._clauses):
            return self._clauses[self._clause_index]
        return None

    def _advance_clause_if_current(self, clause_id: str) -> None:
        """Move to next clause if the acted-upon clause is the current one."""
        current = self._current_clause()
        if current and current.clause_id == clause_id:
            self._clause_index = min(self._clause_index + 1, len(self._clauses))

    def _make_observation(self) -> Observation:
        reviewed_ids = set(
            [i.clause_id for i in self._identified_issues]
            + self._approved_clauses
            + self._skipped_clauses
        )
        remaining = sum(
            1 for c in self._clauses if c.clause_id not in reviewed_ids
        )
        progress = ReviewProgress(
            total_clauses=len(self._clauses),
            reviewed_clauses=len(reviewed_ids),
            approved_clauses=len(self._approved_clauses),
            flagged_clauses=len(set(i.clause_id for i in self._identified_issues)),
            skipped_clauses=len(self._skipped_clauses),
        )
        return Observation(
            task_id=self._task_id,
            document_title=self._task.document_title,
            current_clause=self._current_clause(),
            remaining_clauses=remaining,
            issues_found=list(self._identified_issues),
            progress=progress,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            clarifications=list(self._clarifications),
            hints=self._task.hints if self._step_count == 0 else [],
        )
