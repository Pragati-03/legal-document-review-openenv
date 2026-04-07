"""
Typed Pydantic models for the Legal Document Review OpenEnv environment.
Observation, Action, and Reward models following the OpenEnv specification.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All valid action types an agent can take."""
    IDENTIFY_ISSUE     = "identify_issue"       # Flag a clause as problematic
    SUGGEST_REVISION   = "suggest_revision"     # Propose revised clause text
    APPROVE_CLAUSE     = "approve_clause"       # Mark a clause as acceptable
    REQUEST_CLARIFICATION = "request_clarification"  # Ask for more context
    SUBMIT_REVIEW      = "submit_review"        # Finalise and submit the review
    SKIP_CLAUSE        = "skip_clause"          # Explicitly skip (penalised)


class IssueSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class IssueCategory(str, Enum):
    AMBIGUOUS_LANGUAGE    = "ambiguous_language"
    MISSING_CLAUSE        = "missing_clause"
    UNFAIR_TERM           = "unfair_term"
    COMPLIANCE_RISK       = "compliance_risk"
    INCONSISTENCY         = "inconsistency"
    JURISDICTION_ISSUE    = "jurisdiction_issue"
    LIABILITY_EXPOSURE    = "liability_exposure"
    IP_RISK               = "ip_risk"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Clause(BaseModel):
    clause_id: str
    section: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IdentifiedIssue(BaseModel):
    clause_id: str
    category: IssueCategory
    severity: IssueSeverity
    description: str
    suggested_revision: Optional[str] = None


class ReviewProgress(BaseModel):
    total_clauses: int
    reviewed_clauses: int
    approved_clauses: int
    flagged_clauses: int
    skipped_clauses: int


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    document_title: str
    current_clause: Optional[Clause]
    remaining_clauses: int
    issues_found: List[IdentifiedIssue] = Field(default_factory=list)
    progress: ReviewProgress
    step_count: int
    max_steps: int
    clarifications: List[str] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)


class Action(BaseModel):
    """An action the agent takes on the current clause."""
    action_type: ActionType
    clause_id: Optional[str] = None
    issue_category: Optional[IssueCategory] = None
    issue_severity: Optional[IssueSeverity] = None
    issue_description: Optional[str] = None
    suggested_revision: Optional[str] = None
    clarification_question: Optional[str] = None
    reasoning: Optional[str] = None   # Chain-of-thought (not scored, informational)


class Reward(BaseModel):
    """Partial reward signal returned after each step."""
    step_reward: float = Field(ge=-1.0, le=1.0, description="Reward for this step")
    cumulative_reward: float = Field(description="Total reward so far this episode")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Component breakdown: accuracy, coverage, efficiency, penalty"
    )
    feedback: str = Field(default="", description="Human-readable reward explanation")
