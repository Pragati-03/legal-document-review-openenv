
---
title: Legal Document Review OpenEnv
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
app_port:7860
pinned: false
---



# ⚖️ Legal Document Review — OpenEnv Environment

[![openenv](https://img.shields.io/badge/openenv-compatible-blue)](https://openenv.ai)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-legal--document--review-yellow)](https://huggingface.co/spaces/openenv/legal-document-review)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> An OpenEnv-compliant AI agent environment that simulates **professional legal contract review** — one of the highest-value, most cognitively demanding tasks performed by knowledge workers every day.



## 📋 Overview

Legal contract review is a multi-billion-dollar professional services category. Senior lawyers and paralegals spend thousands of hours annually reading contracts clause by clause, identifying risky or ambiguous language, and suggesting revisions. This environment challenges an AI agent to do the same.

An agent receives a realistic contract document and must:
1. **Read** each clause carefully
2. **Classify** clauses as acceptable or problematic
3. **Identify** the issue category and severity
4. **Describe** the problem in precise legal terms
5. **Suggest** concrete revised language
6. **Submit** a complete review

The environment provides dense, partial reward signals at every step and a comprehensive final score based on detection rate, issue quality, false-positive rate, and efficiency.



## 🗂️ Environment Structure

```
legal-review-env/
├── env/
│   ├── models.py          # Typed Pydantic models (Observation, Action, Reward)
│   └── environment.py     # LegalReviewEnv class — reset(), step(), state()
├── tasks/
│   └── task_definitions.py # 3 tasks with ground-truth annotations
├── graders/
│   └── grader.py          # Deterministic graders, step-level reward shaping
├── server.py              # FastAPI HTTP server (OpenEnv API)
├── baseline_inference.py  # OpenAI-based baseline agent
├── openenv.yaml           # OpenEnv metadata
├── Dockerfile             # Container build for HF Spaces + local use
├── requirements.txt
└── README.md
```



## 🔌 OpenEnv API

### `POST /reset`
Resets the environment and returns the initial `Observation`.

```json
{
  "task_id": "task_easy_freelance",
  "seed": 42
}
```

### `POST /step`
Takes an `Action` and returns `(Observation, Reward, done, info)`.

```json
{
  "action_type": "identify_issue",
  "clause_id": "E03",
  "issue_category": "missing_clause",
  "issue_severity": "critical",
  "issue_description": "Payment amount is unspecified; verbal agreements are unenforceable",
  "suggested_revision": "Fixed fee of $X payable by invoice within 30 days of completion",
  "reasoning": "Clause 3 fails to define any payment amount, making the contract unenforceable on its most essential commercial term."
}
```

### `GET /state`
Returns the full serialisable environment state.



## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Active task identifier |
| `document_title` | string | Name of the contract document |
| `current_clause` | object | Clause currently under review (`clause_id`, `section`, `text`) |
| `remaining_clauses` | int | Number of unreviewed clauses |
| `issues_found` | list | All issues identified so far |
| `progress` | object | `total_clauses`, `reviewed`, `approved`, `flagged`, `skipped` |
| `step_count` | int | Steps taken this episode |
| `max_steps` | int | Step budget for this task |
| `clarifications` | list | Any clarifications the agent has requested |
| `hints` | list | Task hints (shown on first step only) |



## 🎮 Action Space

| `action_type` | Required fields | Description |
|---|---|---|
| `identify_issue` | `clause_id`, `issue_category`, `issue_severity`, `issue_description` | Flag a clause as problematic |
| `suggest_revision` | `clause_id`, `suggested_revision` | Propose revised clause text |
| `approve_clause` | `clause_id` | Mark a clause as acceptable |
| `request_clarification` | `clarification_question` | Ask for more context |
| `submit_review` | — | Finalise and submit the review |
| `skip_clause` | `clause_id` | Skip a clause (penalised) |

**Issue Categories:** `ambiguous_language` · `missing_clause` · `unfair_term` · `compliance_risk` · `inconsistency` · `jurisdiction_issue` · `liability_exposure` · `ip_risk`

**Severities:** `low` · `medium` · `high` · `critical`



## 🏆 Reward Design

### Step-level (dense, partial signals)
Every action receives an immediate reward in `[-0.3, +0.35]`:

| Action | Clause type | Reward |
|---|---|---|
| `identify_issue` | True positive (correct GT clause) | `+0.10` base, `+0.20` for good description, `+0.05` for good revision |
| `approve_clause` | Safe clause | `+0.05` |
| `approve_clause` | GT issue clause | `-0.20` (missed issue) |
| `identify_issue` | Safe clause (false positive) | `-0.10` |
| `identify_issue` | Duplicate (already flagged) | `-0.05` |
| `skip_clause` | Any | `-0.05` |
| `submit_review` | Any | `+0.05` |
| `request_clarification` | Any | `+0.02` |

### Episode-level bonus (at termination)
A final grade is computed and mapped to `[-0.25, +0.25]`, added to cumulative reward.

### Final Score (0.0 – 1.0)
| Component | Weight | Description |
|---|---|---|
| Detection rate | 40% | Fraction of GT issues correctly flagged |
| Issue quality | 30% | Category + severity + description + revision accuracy |
| False-positive rate | 15% | Penalises flagging safe clauses |
| Efficiency | 10% | Fewer steps = higher score |
| Submission | 5% | Completed review submitted |



## 📚 Tasks

### Task 1 — Easy: Freelance Web Development Agreement
**Document:** 8 clauses · **Max steps:** 30 · **GT issues:** 4

A short freelance contract with obvious, beginner-level issues that any competent reviewer should catch:

| Clause | Category | Severity | Issue |
|---|---|---|---|
| E02 (Scope) | `ambiguous_language` | HIGH | Deliverables and timeline undefined |
| E03 (Payment) | `missing_clause` | CRITICAL | No payment amount or method specified |
| E04 (IP) | `ip_risk` | HIGH | Developer retains IP instead of client |
| E06 (Termination) | `unfair_term` | HIGH | One-sided termination, no payment for completed work |

**Baseline (gpt-4o): 0.68** | **Oracle: 0.97**



### Task 2 — Medium: Enterprise SaaS Subscription Agreement
**Document:** 11 clauses · **Max steps:** 50 · **GT issues:** 5

An enterprise SaaS contract with subtler, multi-layered issues requiring domain knowledge:

| Clause | Category | Severity | Issue |
|---|---|---|---|
| M03 (Fees) | `unfair_term` | HIGH | Auto-renewal + 15% price increase, 90-day notice trap |
| M05 (Data) | `compliance_risk` | HIGH | No GDPR/DPA; no Article 28 processor agreement |
| M07 (IP) | `unfair_term` | MEDIUM | Unilateral right to use customer's name/logo for marketing |
| M08 (Indemnity) | `liability_exposure` | HIGH | Asymmetric indemnification — customer exposed, vendor limited |
| M09 (Liability) | `liability_exposure` | HIGH | Liability cap of 3 months is inadequate for enterprise SaaS |

**Baseline (gpt-4o): 0.52** | **Oracle: 0.97**

---

### Task 3 — Hard: M&A Asset Purchase Agreement
**Document:** 12 clauses · **Max steps:** 80 · **GT issues:** 6

A complex M&A transaction agreement requiring cross-clause reasoning, M&A domain expertise, and the ability to spot subtle manipulation risks:

| Clause | Category | Severity | Issue |
|---|---|---|---|
| H03 (WC Adj.) | `ambiguous_language` | HIGH | Buyer prepares working capital adjustment unilaterally, no neutral mechanism |
| H04 (IP Reps) | `ip_risk` | HIGH | IP warranty limited to registered IP only; unregistered IP not covered |
| H06 (Earn-Out) | `ambiguous_language` | CRITICAL | Earn-out manipulation: buyer has operational discretion that conflicts with good faith |
| H07 (Non-Compete) | `unfair_term` | HIGH | 5-year worldwide non-compete including future Buyer products — overbroad and likely unenforceable |
| H08 (Tax) | `liability_exposure` | MEDIUM | Seller deemed to consent to Buyer's tax allocation after only 15 days |
| H09 (Indemnity) | `liability_exposure` | HIGH | 10% indemnification cap and $500k basket are inadequate for a $45M deal |

**Baseline (gpt-4o): 0.31** | **Oracle: 0.97**

> The hard task is designed to challenge state-of-the-art models. GPT-4o consistently misses the earn-out manipulation risk (H06) and the unregistered IP warranty gap (H04), scoring well below a competent M&A associate.



## 🚀 Setup & Usage

### Local (Python)

```bash
git clone https://github.com/openenv/legal-document-review
cd legal-document-review

pip install -r requirements.txt

# Start the API server
python server.py
# Server available at http://localhost:7860
```

### Local (Docker)

```bash
docker build -t legal-review-openenv .
docker run -p 7860:7860 legal-review-openenv
```

### Run baseline agent

```bash
export OPENAI_API_KEY="sk-..."
python baseline_inference.py --model gpt-4o
```

### Python API (no server)

```python
from env.environment import LegalReviewEnv
from env.models import Action, ActionType, IssueCategory, IssueSeverity

env = LegalReviewEnv(task_id="task_easy_freelance")
obs = env.reset()

action = Action(
    action_type=ActionType.IDENTIFY_ISSUE,
    clause_id="E03",
    issue_category=IssueCategory.MISSING_CLAUSE,
    issue_severity=IssueSeverity.CRITICAL,
    issue_description="Payment amount is unspecified; verbal agreements are unenforceable",
    suggested_revision="Fixed fee of $X, payable by invoice within 30 days of delivery",
)

obs, reward, done, info = env.step(action)
print(f"Reward: {reward.step_reward}, Feedback: {reward.feedback}")
```

### HTTP API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_medium_saas"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "identify_issue",
    "clause_id": "M03",
    "issue_category": "unfair_term",
    "issue_severity": "high",
    "issue_description": "Auto-renewal clause with 90-day notice and 15% price increase",
    "suggested_revision": "Reduce notice to 30 days and require written consent for any fee increase"
  }'

# State
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

---

## 📊 Baseline Scores (gpt-4o, seed=42, temperature=0)

| Task | Difficulty | Steps | Detection | Quality | Score |
|---|---|---|---|---|---|
| `task_easy_freelance` | Easy | ~10 | 0.75 | 0.68 | **0.68** |
| `task_medium_saas` | Medium | ~13 | 0.60 | 0.52 | **0.52** |
| `task_hard_ma` | Hard | ~14 | 0.50 | 0.35 | **0.31** |

Scores are reproducible with `OPENAI_API_KEY` set, `temperature=0`, `seed=42`.

---

## 🌍 Real-World Relevance

- **Legal tech** companies (Harvey, Ironclad, Kira) solve exactly this task commercially
- **In-house legal teams** spend 30–60% of time on contract review
- **Procurement** organisations review thousands of supplier contracts annually
- **Regulatory compliance** (GDPR, SOX, employment law) requires clause-level analysis
- Evaluation framework transfers to **due diligence**, **lease review**, **employment agreements**

---

## 🏗️ Hugging Face Spaces Deployment

This environment is deployed at: `https://huggingface.co/spaces/openenv/legal-document-review`

The space uses the `Dockerfile` at the repository root. HF Spaces automatically builds
and serves the container, exposing the OpenEnv HTTP API on port 7860.

Tags: `openenv`, `legal`, `nlp`, `contract-review`, `professional`

