"""
Deterministic graders for each task.
Each grader returns a score in [0.0, 1.0] and a breakdown dict.
Graders are pure functions — given the same inputs they always return the same output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from env.models import IssueCategory, IssueSeverity, IdentifiedIssue
from tasks.task_definitions import GroundTruthIssue, TaskDefinition


@dataclass
class GraderResult:
    score: float                          # 0.0 – 1.0
    breakdown: Dict[str, float]           # sub-scores
    explanation: str


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _keyword_match(text: str, keywords: List[str], threshold: int = 1) -> bool:
    """Return True if at least `threshold` keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits >= threshold


def _severity_score(predicted: IssueSeverity, expected: IssueSeverity) -> float:
    """
    Score how close the predicted severity is to expected.
    Exact match → 1.0; one level off → 0.5; two levels → 0.25; three+ → 0.0
    """
    order = [IssueSeverity.LOW, IssueSeverity.MEDIUM, IssueSeverity.HIGH, IssueSeverity.CRITICAL]
    pi = order.index(predicted)
    ei = order.index(expected)
    diff = abs(pi - ei)
    return {0: 1.0, 1: 0.5, 2: 0.25}.get(diff, 0.0)


def _category_score(predicted: IssueCategory, expected: IssueCategory) -> float:
    return 1.0 if predicted == expected else 0.0


# ---------------------------------------------------------------------------
# Core grader
# ---------------------------------------------------------------------------

def grade_episode(
    task: TaskDefinition,
    identified_issues: List[IdentifiedIssue],
    approved_clauses: List[str],
    skipped_clauses: List[str],
    step_count: int,
    submitted: bool,
) -> GraderResult:
    """
    Universal grader used by all three tasks.
    Returns GraderResult with score in [0.0, 1.0].

    Scoring components
    ------------------
    1. True-positive detection rate   (40%): fraction of ground-truth issues detected
    2. Issue quality                  (30%): category + severity + description accuracy
    3. False-positive penalty         (15%): penalise flagging genuinely safe clauses
    4. Efficiency bonus               (10%): using fewer steps is rewarded
    5. Submission bonus               ( 5%): submitted review vs abandoned

    All components are normalised to [0, 1] before weighting.
    """
    gt_issues: List[GroundTruthIssue] = task.ground_truth_issues
    safe_clause_ids = set(task.safe_clauses)

    # ------ 1. True-positive detection -------
    detected_clause_ids = {i.clause_id for i in identified_issues}
    gt_clause_ids = {g.clause_id for g in gt_issues}

    tp = len(detected_clause_ids & gt_clause_ids)
    detection_rate = tp / max(len(gt_clause_ids), 1)

    # ------ 2. Quality of detected issues -------
    quality_scores: List[float] = []
    for gt in gt_issues:
        # find best matching agent issue for this ground-truth clause
        candidates = [i for i in identified_issues if i.clause_id == gt.clause_id]
        if not candidates:
            quality_scores.append(0.0)
            continue

        best_quality = 0.0
        for candidate in candidates:
            cat_score = _category_score(candidate.category, gt.category)
            sev_score = _severity_score(candidate.severity, gt.severity)
            desc_score = 0.0
            if candidate.description:
                full_kw_match = _keyword_match(
                    candidate.description, gt.description_keywords, threshold=2
                )
                partial_kw_match = _keyword_match(
                    candidate.description, gt.partial_credit_keywords, threshold=1
                )
                desc_score = 1.0 if full_kw_match else (0.5 if partial_kw_match else 0.0)

            revision_score = 0.0
            if candidate.suggested_revision and gt.acceptable_revisions:
                revision_score = 1.0 if any(
                    rev.lower() in candidate.suggested_revision.lower()
                    for rev in gt.acceptable_revisions
                ) else 0.0

            # Weighted quality for this candidate
            q = 0.3 * cat_score + 0.25 * sev_score + 0.3 * desc_score + 0.15 * revision_score
            best_quality = max(best_quality, q)

        quality_scores.append(best_quality)

    issue_quality = sum(quality_scores) / max(len(quality_scores), 1)

    # ------ 3. False-positive penalty -------
    false_positives = len(
        [i for i in identified_issues if i.clause_id in safe_clause_ids]
    )
    fp_penalty = min(false_positives / max(len(safe_clause_ids), 1), 1.0)
    fp_score = 1.0 - fp_penalty   # high score = few false positives

    # ------ 4. Efficiency -------
    efficiency = max(0.0, 1.0 - (step_count / task.max_steps))

    # ------ 5. Submission bonus -------
    submission = 1.0 if submitted else 0.0

    # ------ Weighted total -------
    total = (
        0.40 * detection_rate
        + 0.30 * issue_quality
        + 0.15 * fp_score
        + 0.10 * efficiency
        + 0.05 * submission
    )

    breakdown = {
        "detection_rate":  round(detection_rate, 4),
        "issue_quality":   round(issue_quality, 4),
        "fp_score":        round(fp_score, 4),
        "efficiency":      round(efficiency, 4),
        "submission":      round(submission, 4),
        "total":           round(total, 4),
    }

    explanation = (
        f"Detected {tp}/{len(gt_clause_ids)} issues "
        f"(detection={detection_rate:.2f}, quality={issue_quality:.2f}, "
        f"fp_score={fp_score:.2f}, efficiency={efficiency:.2f}). "
        f"Total score: {total:.4f}"
    )

    return GraderResult(score=round(min(max(total, 0.0), 1.0), 4), breakdown=breakdown, explanation=explanation)


# ---------------------------------------------------------------------------
# Step-level partial reward (called on every step for shaping)
# ---------------------------------------------------------------------------

def step_reward(
    task: TaskDefinition,
    action_type: str,
    clause_id: str,
    issue_category: str | None,
    issue_severity: str | None,
    issue_description: str | None,
    suggested_revision: str | None,
    cumulative_issues: List[IdentifiedIssue],
    step_count: int,
) -> Tuple[float, str]:
    """
    Returns (reward: float, feedback: str) for a single step.
    Reward is in [-0.3, 0.3] — final episode score is separate.
    """
    gt_clause_ids = {g.clause_id for g in task.ground_truth_issues}
    safe_clause_ids = set(task.safe_clauses)

    if action_type == "skip_clause":
        return (-0.05, "Skipping clauses reduces coverage — review everything.")

    if action_type == "approve_clause":
        if clause_id in gt_clause_ids:
            return (-0.2, f"Clause {clause_id} has issues — approving it is incorrect.")
        if clause_id in safe_clause_ids:
            return (0.05, f"Clause {clause_id} correctly approved as safe.")
        return (0.0, "Approval recorded.")

    if action_type == "identify_issue":
        if clause_id in safe_clause_ids:
            return (-0.1, f"Clause {clause_id} appears to be acceptable — re-check.")
        if clause_id in gt_clause_ids:
            already_found = [i for i in cumulative_issues if i.clause_id == clause_id]
            if already_found:
                return (-0.05, f"Issue in {clause_id} already identified — avoid duplicates.")
            gt = next(g for g in task.ground_truth_issues if g.clause_id == clause_id)
            desc_score = 0.0
            if issue_description:
                if _keyword_match(issue_description, gt.description_keywords, threshold=2):
                    desc_score = 0.2
                elif _keyword_match(issue_description, gt.partial_credit_keywords, threshold=1):
                    desc_score = 0.1
            base = 0.1
            rev_bonus = 0.05 if suggested_revision and any(
                r.lower() in (suggested_revision or "").lower()
                for r in gt.acceptable_revisions
            ) else 0.0
            total = base + desc_score + rev_bonus
            return (round(total, 3), f"Issue detected in {clause_id} — good catch!")
        return (-0.05, f"Clause {clause_id} not in the known problematic set.")

    if action_type == "suggest_revision":
        if clause_id in gt_clause_ids:
            gt = next(g for g in task.ground_truth_issues if g.clause_id == clause_id)
            if suggested_revision and any(
                r.lower() in suggested_revision.lower()
                for r in gt.acceptable_revisions
            ):
                return (0.15, f"Revision for {clause_id} is on the right track.")
            return (0.05, f"Revision noted but may not fully address the issue in {clause_id}.")
        return (0.0, "Revision recorded.")

    if action_type == "request_clarification":
        return (0.02, "Clarification request recorded.")

    if action_type == "submit_review":
        return (0.05, "Review submitted.")

    return (0.0, "Action recorded.")
