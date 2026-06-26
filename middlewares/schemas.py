"""Pydantic schemas for the QueueStorm Investigator API.

Defines the request body for POST /analyze-ticket and the response body it must
return, per the rubric's Sections 5 and 6. Literal type aliases for the enum
fields live here too so the taxonomy is colocated with the models that use it.
"""

from typing import Optional, List, Literal

from pydantic import BaseModel, Field, field_validator


# ── Request enums ──────────────────────────────────────────────────────────

Channel = Literal["in_app_chat", "call_center", "email", "merchant_portal", "field_agent"]

Language = Literal["en", "bn", "mixed"]

UserType = Literal["customer", "merchant", "agent", "unknown"]

TransactionType = Literal["transfer", "payment", "cash_in", "cash_out", "settlement", "refund"]

TransactionStatus = Literal["completed", "failed", "pending", "reversed"]


# ── Response enums ─────────────────────────────────────────────────────────

EvidenceVerdict = Literal["consistent", "inconsistent", "insufficient_data"]

CaseType = Literal[
    "wrong_transfer",
    "payment_failed",
    "refund_request",
    "duplicate_payment",
    "merchant_settlement_delay",
    "agent_cash_in_issue",
    "phishing_or_social_engineering",
    "other",
]

Severity = Literal["low", "medium", "high", "critical"]

Department = Literal[
    "customer_support",
    "dispute_resolution",
    "payments_ops",
    "merchant_operations",
    "agent_operations",
    "fraud_risk",
]


# ── Models ─────────────────────────────────────────────────────────────────

class TransactionHistoryEntry(BaseModel):
    """A single transaction in the customer's recent history."""
    transaction_id: str = Field(..., description="Unique transaction identifier.")
    timestamp: str = Field(..., description="ISO 8601 timestamp when the transaction occurred.")
    type: TransactionType = Field(..., description="Type of transaction.")
    amount: float = Field(..., description="Amount in BDT.")
    counterparty: str = Field(..., description="Recipient phone number, merchant ID, or agent ID.")
    status: TransactionStatus = Field(..., description="Status of the transaction.")


class AnalyzeTicketRequest(BaseModel):
    """Request body for POST /analyze-ticket."""
    ticket_id: str = Field(..., description="Unique ticket identifier. Must be echoed in the response.")
    complaint: str = Field(
        ...,
        min_length=1,
        description="Customer complaint text in English, Bangla, or mixed Banglish.",
    )
    language: Optional[Language] = Field(
        default=None,
        description="One of: en, bn, mixed.",
    )
    channel: Optional[Channel] = Field(
        default=None,
        description="One of: in_app_chat, call_center, email, merchant_portal, field_agent.",
    )
    user_type: Optional[UserType] = Field(
        default=None,
        description="One of: customer, merchant, agent, unknown.",
    )
    campaign_context: Optional[str] = Field(
        default=None,
        description="Campaign identifier provided by the harness.",
    )
    transaction_history: Optional[List[TransactionHistoryEntry]] = Field(
        default=None,
        description="List of recent transactions (typically 2 to 5 entries). May be empty for safety-only cases.",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional simulated context provided by the harness.",
    )

    @field_validator("complaint")
    @classmethod
    def complaint_not_empty(cls, v: str) -> str:
        # Pydantic ValueError → mapped to 422 by the RequestValidationError handler
        # in exceptions.py.
        if not v.strip():
            raise ValueError("complaint must not be empty or whitespace only")
        return v

    @field_validator("transaction_history")
    @classmethod
    def transaction_history_valid(cls, v):
        if v is not None and len(v) > 20:
            raise ValueError("transaction_history must contain at most 20 entries")
        return v


class AnalyzeTicketResponse(BaseModel):
    """Response body for POST /analyze-ticket."""
    ticket_id: str = Field(..., description="Must match the value sent in the request.")
    relevant_transaction_id: Optional[str] = Field(
        ...,
        description="Transaction ID the complaint refers to, or null if none in the provided history matches.",
    )
    evidence_verdict: EvidenceVerdict = Field(
        ...,
        description="One of: consistent, inconsistent, insufficient_data.",
    )
    verdict_reason: str = Field(
        ...,
        description="Reasoning behind the evidence_verdict, explaining why the matched transaction supports, contradicts, or is insufficient to evaluate the complaint.",
    )
    case_type: CaseType = Field(..., description="From the taxonomy in Section 7.1.")
    severity: Severity = Field(..., description="One of: low, medium, high, critical.")
    department: Department = Field(..., description="From the taxonomy in Section 7.2.")
    agent_summary: str = Field(
        ...,
        min_length=1,
        description="Concise agent-ready summary of the case (one to two sentences).",
    )
    recommended_next_action: str = Field(
        ...,
        min_length=1,
        description="Suggested operational next step for the support agent.",
    )
    customer_reply: str = Field(
        ...,
        min_length=1,
        description="Safe official reply that respects all safety rules in Section 8.",
    )
    human_review_required: bool = Field(
        ...,
        description="True for disputes, suspicious cases, high-value cases, or ambiguous evidence.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Float between 0 and 1.",
    )
    reason_codes: Optional[List[str]] = Field(
        default=None,
        description="Short reason labels supporting the decision.",
    )