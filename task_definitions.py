"""
Three real-world legal document review tasks with increasing difficulty.
Each task contains a realistic contract document and ground-truth annotations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from env.models import IssueCategory, IssueSeverity


@dataclass
class GroundTruthIssue:
    clause_id: str
    category: IssueCategory
    severity: IssueSeverity
    description_keywords: List[str]          # Keywords that must appear in agent description
    acceptable_revisions: List[str] = field(default_factory=list)  # Acceptable revision patterns
    partial_credit_keywords: List[str] = field(default_factory=list)


@dataclass
class TaskDefinition:
    task_id: str
    difficulty: str                          # easy / medium / hard
    document_title: str
    description: str
    clauses: List[Dict[str, Any]]
    ground_truth_issues: List[GroundTruthIssue]
    safe_clauses: List[str]                  # clause_ids that are genuinely fine
    max_steps: int
    hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# TASK 1 — EASY: Simple Freelance Service Agreement
# ===========================================================================

TASK_EASY = TaskDefinition(
    task_id="task_easy_freelance",
    difficulty="easy",
    document_title="Freelance Web Development Service Agreement",
    description=(
        "A short freelance contract between a web developer and a small business. "
        "Identify obvious problematic clauses such as missing payment terms, vague IP ownership, "
        "and one-sided termination clauses."
    ),
    clauses=[
        {
            "clause_id": "E01",
            "section": "1. Parties",
            "text": (
                "This Agreement is between DevCo ('Developer') and AcmeCorp ('Client'), "
                "entered into as of the date last signed below."
            ),
        },
        {
            "clause_id": "E02",
            "section": "2. Scope of Work",
            "text": (
                "Developer agrees to build a website for Client. "
                "Exact deliverables, timelines, and acceptance criteria will be determined later."
            ),
        },
        {
            "clause_id": "E03",
            "section": "3. Payment",
            "text": (
                "Client will pay Developer a reasonable amount upon completion. "
                "Payment method and schedule are to be agreed verbally."
            ),
        },
        {
            "clause_id": "E04",
            "section": "4. Intellectual Property",
            "text": (
                "All work product created under this Agreement, including source code, designs, "
                "and documentation, shall be owned exclusively by the Developer upon delivery, "
                "and Client receives a non-exclusive licence to use the work."
            ),
        },
        {
            "clause_id": "E05",
            "section": "5. Confidentiality",
            "text": (
                "Each party agrees to keep confidential all non-public information received "
                "from the other party during the term of this Agreement and for two years thereafter."
            ),
        },
        {
            "clause_id": "E06",
            "section": "6. Termination",
            "text": (
                "Client may terminate this Agreement at any time without notice and without any "
                "obligation to pay for work already completed."
            ),
        },
        {
            "clause_id": "E07",
            "section": "7. Governing Law",
            "text": (
                "This Agreement shall be governed by the laws of the State of Delaware, USA."
            ),
        },
        {
            "clause_id": "E08",
            "section": "8. Entire Agreement",
            "text": (
                "This Agreement constitutes the entire agreement between the parties and supersedes "
                "all prior negotiations, representations, or agreements, whether written or oral."
            ),
        },
    ],
    ground_truth_issues=[
        GroundTruthIssue(
            clause_id="E02",
            category=IssueCategory.AMBIGUOUS_LANGUAGE,
            severity=IssueSeverity.HIGH,
            description_keywords=["scope", "deliverable", "vague", "undefined", "timeline"],
            acceptable_revisions=["specific deliverables", "milestone", "acceptance criteria"],
            partial_credit_keywords=["unclear", "ambiguous", "determined later"],
        ),
        GroundTruthIssue(
            clause_id="E03",
            category=IssueCategory.MISSING_CLAUSE,
            severity=IssueSeverity.CRITICAL,
            description_keywords=["payment", "amount", "fee", "verbal", "undefined"],
            acceptable_revisions=["fixed fee", "hourly rate", "invoice", "payment schedule"],
            partial_credit_keywords=["unspecified", "vague", "reasonable"],
        ),
        GroundTruthIssue(
            clause_id="E04",
            category=IssueCategory.IP_RISK,
            severity=IssueSeverity.HIGH,
            description_keywords=["IP", "intellectual property", "ownership", "developer", "client"],
            acceptable_revisions=["assign", "transfer", "client owns", "work for hire"],
            partial_credit_keywords=["non-exclusive", "licence", "unusual"],
        ),
        GroundTruthIssue(
            clause_id="E06",
            category=IssueCategory.UNFAIR_TERM,
            severity=IssueSeverity.HIGH,
            description_keywords=["termination", "no notice", "without pay", "one-sided"],
            acceptable_revisions=["notice period", "pro-rata", "completed work payment"],
            partial_credit_keywords=["unfair", "developer", "compensation"],
        ),
    ],
    safe_clauses=["E01", "E05", "E07", "E08"],
    max_steps=30,
    hints=[
        "Check payment terms for specificity.",
        "Consider who should own the IP after delivery.",
        "Review whether termination is fair to both parties.",
    ],
)


# ===========================================================================
# TASK 2 — MEDIUM: SaaS Subscription Agreement
# ===========================================================================

TASK_MEDIUM = TaskDefinition(
    task_id="task_medium_saas",
    difficulty="medium",
    document_title="SaaS Enterprise Subscription Agreement",
    description=(
        "An enterprise SaaS contract between a software vendor and a large enterprise customer. "
        "Subtler issues including data processing obligations, SLA limitations, auto-renewal traps, "
        "indemnification asymmetry, and GDPR compliance gaps must be identified."
    ),
    clauses=[
        {
            "clause_id": "M01",
            "section": "1. Definitions",
            "text": (
                "'Service' means the cloud-based software platform made available by Vendor. "
                "'User Data' means any data submitted by Customer or its users to the Service."
            ),
        },
        {
            "clause_id": "M02",
            "section": "2. Subscription & Access",
            "text": (
                "Vendor grants Customer a non-exclusive, non-transferable right to access and use "
                "the Service during the Subscription Term solely for Customer's internal business purposes."
            ),
        },
        {
            "clause_id": "M03",
            "section": "3. Fees & Renewal",
            "text": (
                "Subscription fees are due annually in advance. This Agreement automatically renews "
                "for successive one-year terms unless either party provides written notice of "
                "non-renewal at least 90 days before the end of the then-current term. "
                "Vendor may increase fees by up to 15% upon renewal without additional notice."
            ),
        },
        {
            "clause_id": "M04",
            "section": "4. Service Level Agreement",
            "text": (
                "Vendor targets 99.5% monthly uptime. In the event of downtime exceeding the SLA, "
                "Customer's sole remedy is a service credit equal to 5% of the monthly fee for each "
                "full hour of excess downtime, capped at one month's subscription fee. "
                "Credits will not be issued if downtime results from Customer's actions or third-party services."
            ),
        },
        {
            "clause_id": "M05",
            "section": "5. Data Processing & Privacy",
            "text": (
                "Vendor will process User Data as necessary to provide the Service. "
                "Customer represents that it has obtained all necessary consents for data submitted to the Service. "
                "Vendor may use aggregated, anonymised User Data for product improvement purposes."
            ),
        },
        {
            "clause_id": "M06",
            "section": "6. Data Security",
            "text": (
                "Vendor will implement reasonable technical and organisational measures to protect User Data. "
                "In the event of a confirmed data breach affecting Customer's data, "
                "Vendor will notify Customer within 72 hours of becoming aware of the breach."
            ),
        },
        {
            "clause_id": "M07",
            "section": "7. Intellectual Property",
            "text": (
                "All rights in the Service, including all improvements, modifications, and derivatives, "
                "remain with Vendor. Customer retains all rights to User Data. "
                "Customer grants Vendor a licence to use Customer's name and logo for marketing purposes."
            ),
        },
        {
            "clause_id": "M08",
            "section": "8. Indemnification",
            "text": (
                "Customer shall indemnify and hold harmless Vendor, its officers, directors, and employees "
                "from any claims arising from: (a) Customer's use of the Service; (b) violation of this Agreement; "
                "or (c) infringement of third-party intellectual property rights by Customer. "
                "Vendor's indemnification obligations are limited to direct IP infringement claims only."
            ),
        },
        {
            "clause_id": "M09",
            "section": "9. Limitation of Liability",
            "text": (
                "In no event shall Vendor be liable for any indirect, incidental, special, punitive, "
                "or consequential damages. Vendor's total cumulative liability shall not exceed "
                "the fees paid by Customer in the three months preceding the claim."
            ),
        },
        {
            "clause_id": "M10",
            "section": "10. Termination",
            "text": (
                "Either party may terminate for material breach if the breach remains uncured "
                "30 days after written notice. Upon termination, Customer's access ceases immediately "
                "and all fees paid are non-refundable. Vendor will delete Customer data within 90 days of termination."
            ),
        },
        {
            "clause_id": "M11",
            "section": "11. Governing Law & Dispute Resolution",
            "text": (
                "This Agreement shall be governed by the laws of the State of California. "
                "Any disputes shall be resolved by binding arbitration under AAA rules, "
                "with arbitration to be held in San Francisco, California."
            ),
        },
    ],
    ground_truth_issues=[
        GroundTruthIssue(
            clause_id="M03",
            category=IssueCategory.UNFAIR_TERM,
            severity=IssueSeverity.HIGH,
            description_keywords=["auto-renewal", "90 days", "15%", "price increase", "notice"],
            acceptable_revisions=["30 days", "60 days", "no automatic increase", "cap", "consent"],
            partial_credit_keywords=["renewal", "fee increase", "automatic"],
        ),
        GroundTruthIssue(
            clause_id="M05",
            category=IssueCategory.COMPLIANCE_RISK,
            severity=IssueSeverity.HIGH,
            description_keywords=["GDPR", "data processing agreement", "DPA", "personal data", "processor"],
            acceptable_revisions=["data processing agreement", "DPA", "GDPR Article 28", "controller", "processor"],
            partial_credit_keywords=["privacy", "regulation", "consent", "data protection"],
        ),
        GroundTruthIssue(
            clause_id="M07",
            category=IssueCategory.UNFAIR_TERM,
            severity=IssueSeverity.MEDIUM,
            description_keywords=["logo", "marketing", "consent", "unilateral", "brand"],
            acceptable_revisions=["prior written consent", "opt-out", "approval", "remove"],
            partial_credit_keywords=["name use", "marketing rights"],
        ),
        GroundTruthIssue(
            clause_id="M08",
            category=IssueCategory.LIABILITY_EXPOSURE,
            severity=IssueSeverity.HIGH,
            description_keywords=["indemnification", "asymmetric", "one-sided", "vendor", "limited"],
            acceptable_revisions=["mutual indemnification", "IP infringement by vendor", "reciprocal"],
            partial_credit_keywords=["unbalanced", "customer only", "broad"],
        ),
        GroundTruthIssue(
            clause_id="M09",
            category=IssueCategory.LIABILITY_EXPOSURE,
            severity=IssueSeverity.HIGH,
            description_keywords=["liability cap", "three months", "inadequate", "low", "damages"],
            acceptable_revisions=["12 months", "annual fees", "higher cap", "12-month"],
            partial_credit_keywords=["cap", "too low", "insufficient", "consequential"],
        ),
    ],
    safe_clauses=["M01", "M02", "M04", "M06", "M10", "M11"],
    max_steps=50,
    hints=[
        "Consider GDPR/data protection obligations when reviewing data clauses.",
        "Evaluate whether indemnification obligations are mutual.",
        "Check whether liability caps adequately protect the customer.",
    ],
)


# ===========================================================================
# TASK 3 — HARD: M&A Asset Purchase Agreement (complex)
# ===========================================================================

TASK_HARD = TaskDefinition(
    task_id="task_hard_ma",
    difficulty="hard",
    document_title="Asset Purchase Agreement — Acquisition of TechStartup Inc.",
    description=(
        "A complex M&A asset purchase agreement for the acquisition of a tech startup. "
        "Identify subtle, layered issues including: earn-out manipulation risks, reps & warranties gaps, "
        "material adverse change definition loopholes, non-compete overreach, "
        "tax allocation asymmetry, and IP assignment completeness concerns. "
        "Some issues require reading across multiple clauses simultaneously."
    ),
    clauses=[
        {
            "clause_id": "H01",
            "section": "1. Definitions — Material Adverse Effect",
            "text": (
                "'Material Adverse Effect' (MAE) means any change, event, or circumstance that has had, "
                "or would reasonably be expected to have, a material adverse effect on the business, "
                "financial condition, or results of operations of the Company, "
                "excluding effects resulting from: (i) changes in general economic conditions; "
                "(ii) changes in conditions in the technology industry generally; "
                "(iii) acts of war or terrorism; (iv) changes in applicable law or GAAP. "
                "Notwithstanding the foregoing, no exclusion shall apply to the extent "
                "such change disproportionately affects the Company relative to other participants "
                "in the technology industry."
            ),
        },
        {
            "clause_id": "H02",
            "section": "2. Purchase Price",
            "text": (
                "The aggregate purchase price shall be USD 45,000,000 (the 'Base Purchase Price'), "
                "plus the Earn-Out Payments as defined in Section 6, subject to adjustment "
                "pursuant to Section 3."
            ),
        },
        {
            "clause_id": "H03",
            "section": "3. Purchase Price Adjustment",
            "text": (
                "The Base Purchase Price shall be adjusted based on the Closing Working Capital. "
                "If the Closing Working Capital is less than the Target Working Capital of USD 2,000,000, "
                "the purchase price shall be reduced dollar-for-dollar. "
                "If it exceeds the Target, the purchase price shall be increased dollar-for-dollar. "
                "The Closing Working Capital statement shall be prepared by Buyer in accordance with "
                "the Company's historical accounting practices."
            ),
        },
        {
            "clause_id": "H04",
            "section": "4. Representations & Warranties — Intellectual Property",
            "text": (
                "Seller represents and warrants that: (a) the Company owns or has valid licences to "
                "all Intellectual Property used in the Business; (b) to Seller's Knowledge, "
                "no third party is infringing any Company IP; (c) the Company IP does not infringe "
                "any third-party rights. Section (c) applies only to IP registered as of the Closing Date."
            ),
        },
        {
            "clause_id": "H05",
            "section": "5. Representations & Warranties — Employees",
            "text": (
                "To Seller's Knowledge, no key employee has communicated an intention to resign. "
                "The Company is in material compliance with all applicable employment laws. "
                "There are no pending or, to Seller's Knowledge, threatened labour disputes."
            ),
        },
        {
            "clause_id": "H06",
            "section": "6. Earn-Out",
            "text": (
                "Seller shall be entitled to additional earn-out payments of up to USD 15,000,000 "
                "based on the revenue of the Business for the 24-month period following Closing ('Earn-Out Period'). "
                "Earn-Out Payments will be calculated and paid by Buyer. "
                "During the Earn-Out Period, Buyer shall operate the Business in good faith "
                "and shall not take any action with the primary purpose of reducing Earn-Out Payments. "
                "Buyer shall have sole discretion over business operations, pricing, and resource allocation."
            ),
        },
        {
            "clause_id": "H07",
            "section": "7. Non-Compete",
            "text": (
                "For a period of five (5) years following the Closing Date, Seller and each of its principals "
                "shall not, directly or indirectly, engage in, own, manage, operate, control, or participate in "
                "any business that competes with the Business anywhere in the world. "
                "This restriction applies to any business that could potentially compete with any product "
                "or service that Buyer or any Buyer affiliate currently offers or may offer in the future."
            ),
        },
        {
            "clause_id": "H08",
            "section": "8. Tax Matters",
            "text": (
                "The parties agree to allocate the purchase price among the acquired assets in accordance "
                "with Section 1060 of the Internal Revenue Code. Buyer shall prepare the IRS Form 8594 "
                "asset allocation and shall provide Seller with the proposed allocation within 90 days of Closing. "
                "Seller shall be deemed to consent to the allocation if it does not object within 15 days of receipt. "
                "Buyer shall bear all transfer taxes."
            ),
        },
        {
            "clause_id": "H09",
            "section": "9. Indemnification — Seller",
            "text": (
                "Seller shall indemnify Buyer for losses arising from: (a) any breach of Seller's representations "
                "or warranties; (b) any pre-closing tax liabilities; (c) any excluded liabilities. "
                "Indemnification claims must be made within 18 months of Closing (the 'Survival Period'), "
                "except for Fundamental Representations which survive for 6 years. "
                "Seller's aggregate indemnification liability is capped at 10% of the Base Purchase Price. "
                "A deductible of USD 500,000 applies before any indemnification obligation is triggered."
            ),
        },
        {
            "clause_id": "H10",
            "section": "10. Conditions to Closing — Regulatory",
            "text": (
                "The obligation of Buyer to consummate the transactions is conditioned upon: "
                "(a) obtaining all required regulatory approvals; (b) no governmental entity having enacted "
                "any law that prohibits consummation; (c) no Material Adverse Effect having occurred. "
                "Buyer shall use commercially reasonable efforts to obtain regulatory approvals."
            ),
        },
        {
            "clause_id": "H11",
            "section": "11. IP Assignment",
            "text": (
                "Seller hereby assigns to Buyer all right, title, and interest in and to all Intellectual Property "
                "owned by the Company as of the Closing Date, including patents, trademarks, copyrights, "
                "and trade secrets. Each employee and contractor who has created IP for the Company "
                "shall execute an assignment agreement prior to Closing."
            ),
        },
        {
            "clause_id": "H12",
            "section": "12. Governing Law & Arbitration",
            "text": (
                "This Agreement shall be governed by Delaware law. Disputes shall be resolved by "
                "binding arbitration under JAMS rules, with a single arbitrator, "
                "conducted in New York, New York."
            ),
        },
    ],
    ground_truth_issues=[
        GroundTruthIssue(
            clause_id="H03",
            category=IssueCategory.AMBIGUOUS_LANGUAGE,
            severity=IssueSeverity.HIGH,
            description_keywords=["buyer prepares", "working capital", "one-sided", "seller review", "dispute mechanism"],
            acceptable_revisions=["independent accountant", "neutral arbitrator", "seller review period", "dispute resolution"],
            partial_credit_keywords=["unilateral", "buyer controls", "adjustment"],
        ),
        GroundTruthIssue(
            clause_id="H04",
            category=IssueCategory.IP_RISK,
            severity=IssueSeverity.HIGH,
            description_keywords=["knowledge qualifier", "unregistered IP", "only registered", "limitation", "gap"],
            acceptable_revisions=["remove knowledge qualifier", "all IP", "unregistered", "broader warranty"],
            partial_credit_keywords=["qualified", "seller's knowledge", "registered only"],
        ),
        GroundTruthIssue(
            clause_id="H06",
            category=IssueCategory.AMBIGUOUS_LANGUAGE,
            severity=IssueSeverity.CRITICAL,
            description_keywords=["earn-out", "manipulation", "sole discretion", "conflict", "good faith"],
            acceptable_revisions=["specific operating covenants", "minimum spend", "independent audit", "neutral calculation"],
            partial_credit_keywords=["discretion", "buyer controls", "conflict of interest", "manipulation"],
        ),
        GroundTruthIssue(
            clause_id="H07",
            category=IssueCategory.UNFAIR_TERM,
            severity=IssueSeverity.HIGH,
            description_keywords=["non-compete", "five years", "worldwide", "future products", "overbroad", "enforceability"],
            acceptable_revisions=["three years", "geographic limit", "current products", "narrower scope"],
            partial_credit_keywords=["too broad", "unenforceable", "five year", "worldwide"],
        ),
        GroundTruthIssue(
            clause_id="H08",
            category=IssueCategory.LIABILITY_EXPOSURE,
            severity=IssueSeverity.MEDIUM,
            description_keywords=["tax allocation", "buyer prepares", "15 days", "deemed consent", "seller disadvantaged"],
            acceptable_revisions=["30 days", "negotiated allocation", "mutual agreement", "longer review"],
            partial_credit_keywords=["deemed consent", "unilateral", "short window", "tax"],
        ),
        GroundTruthIssue(
            clause_id="H09",
            category=IssueCategory.LIABILITY_EXPOSURE,
            severity=IssueSeverity.HIGH,
            description_keywords=["10% cap", "low", "indemnification", "basket", "$500,000", "inadequate"],
            acceptable_revisions=["15%", "20%", "higher cap", "lower basket", "remove basket"],
            partial_credit_keywords=["cap too low", "deductible", "insufficient coverage"],
        ),
    ],
    safe_clauses=["H01", "H02", "H05", "H10", "H11", "H12"],
    max_steps=80,
    hints=[
        "Look for conflicts between earn-out protections and buyer's operational discretion.",
        "Consider whether IP representations cover unregistered IP and open source.",
        "Evaluate the indemnification cap relative to deal size and risk profile.",
        "Review non-compete scope for enforceability under modern case law.",
    ],
    metadata={"deal_size_usd": 45_000_000, "jurisdiction": "Delaware", "industry": "Technology"},
)


ALL_TASKS: Dict[str, TaskDefinition] = {
    TASK_EASY.task_id:   TASK_EASY,
    TASK_MEDIUM.task_id: TASK_MEDIUM,
    TASK_HARD.task_id:   TASK_HARD,
}
