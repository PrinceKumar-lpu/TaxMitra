"""
Pydantic Schemas for TaxMitra API requests and responses.
Defines strictly validated data models for /chat, /calculate, and /compare endpoints.
"""

from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field


class DeductionDetails(BaseModel):
    section_80c: float = Field(default=0.0, ge=0.0, description="Investments under 80C (EPF, PPF, ELSS, etc.)")
    section_80d: float = Field(default=0.0, ge=0.0, description="Health insurance premium under 80D")
    section_80ccd_1b: float = Field(default=0.0, ge=0.0, description="Additional NPS tier-1 deposit under 80CCD(1B)")
    hra_exemption: float = Field(default=0.0, ge=0.0, description="Eligible HRA exemption under Section 10(13A)")
    home_loan_interest_24b: float = Field(default=0.0, ge=0.0, description="Self-occupied home loan interest under Section 24(b)")
    other_deductions: float = Field(default=0.0, ge=0.0, description="Other allowable deductions")


class CalculateTaxRequest(BaseModel):
    gross_income: float = Field(gt=0, description="Gross annual salary in INR")
    regime: Literal["new", "old"] = Field(default="new", description="Tax regime choice")
    deductions: Optional[DeductionDetails] = Field(default=None, description="Optional deductions for Old Regime")


class CompareRegimesRequest(BaseModel):
    gross_income: float = Field(gt=0, description="Gross annual salary in INR")
    deductions: Optional[DeductionDetails] = Field(default=None, description="Deductions available in Old Regime")


class SourceCitation(BaseModel):
    source: str = Field(description="Document filename, e.g., 'old_vs_new_regime.md'")
    section: str = Field(description="Section heading inside the document")
    relevance_score: Optional[float] = Field(default=None, description="Cosine similarity score (0.0 to 1.0)")
    snippet: Optional[str] = Field(default=None, description="Excerpt from the source document")


class ToolCallRecord(BaseModel):
    tool_name: str = Field(description="Name of the deterministic tool called (e.g. calculate_tax, compare_regimes)")
    arguments: Dict[str, Any] = Field(description="Arguments passed to the tool")
    output: Dict[str, Any] = Field(description="Result returned by the tool")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="User prompt or tax query")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation and financial profile continuity")


class ChatResponse(BaseModel):
    session_id: str = Field(description="Active session ID")
    answer: str = Field(description="Detailed, markdown-formatted response for the user")
    sources: List[SourceCitation] = Field(default_factory=list, description="List of knowledge base citations")
    tool_calls: List[ToolCallRecord] = Field(default_factory=list, description="Deterministic tool executions made for this turn")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level based on retrieved context and tool accuracy")
    disclaimer: str = Field(
        default="TaxMitra is an AI assistant providing estimates based on Indian Income Tax laws (FY 2024-25 / AY 2025-26). It does not constitute professional chartered accountancy advice.",
        description="Statutory legal and educational disclaimer"
    )
