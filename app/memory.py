"""
Session memory and user financial profile manager.
Maintains:
- Sliding-window chat turns per session_id (e.g. last 6 turns)
- Extracted user profile (income, regime preference, claimed deductions)
This prevents the user from repeatedly typing their salary across conversational turns.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid
import time
import re


class UserFinancialProfile(BaseModel):
    gross_income: Optional[float] = None
    preferred_regime: Optional[str] = None
    section_80c: Optional[float] = None
    section_80d: Optional[float] = None
    hra_exemption: Optional[float] = None
    section_80ccd_1b: Optional[float] = None
    home_loan_interest: Optional[float] = None
    updated_at: float = Field(default_factory=time.time)

    def to_summary_str(self) -> str:
        items = []
        if self.gross_income is not None:
            items.append(f"Gross Income: ₹{self.gross_income:,.0f}")
        if self.preferred_regime:
            items.append(f"Regime Preference: {self.preferred_regime.capitalize()} Regime")
        if self.section_80c is not None:
            items.append(f"80C Investments: ₹{self.section_80c:,.0f}")
        if self.section_80d is not None:
            items.append(f"80D Health Insurance: ₹{self.section_80d:,.0f}")
        if self.hra_exemption is not None:
            items.append(f"HRA: ₹{self.hra_exemption:,.0f}")
        if self.section_80ccd_1b is not None:
            items.append(f"NPS 80CCD(1B): ₹{self.section_80ccd_1b:,.0f}")
        if self.home_loan_interest is not None:
            items.append(f"Home Loan Interest: ₹{self.home_loan_interest:,.0f}")

        if not items:
            return "No known user financial profile yet."
        return ", ".join(items)


class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str
    timestamp: float = Field(default_factory=time.time)


class SessionState(BaseModel):
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    profile: UserFinancialProfile = Field(default_factory=UserFinancialProfile)
    last_active: float = Field(default_factory=time.time)


class MemoryManager:
    """In-memory thread-safe session store with profile extraction heuristic."""

    def __init__(self, max_history_turns: int = 6):
        self._sessions: Dict[str, SessionState] = {}
        self.max_history_turns = max_history_turns

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        
        session = self._sessions[session_id]
        session.last_active = time.time()
        return session

    def add_user_message(self, session_id: str, message: str):
        session = self.get_or_create_session(session_id)
        session.messages.append(ChatMessage(role="user", content=message))
        self._extract_profile_hints(session.profile, message)
        self._trim_history(session)

    def add_model_message(self, session_id: str, message: str):
        session = self.get_or_create_session(session_id)
        session.messages.append(ChatMessage(role="model", content=message))
        self._trim_history(session)

    def _trim_history(self, session: SessionState):
        """Keeps only the most recent N turns to avoid context bloat."""
        max_messages = self.max_history_turns * 2
        if len(session.messages) > max_messages:
            session.messages = session.messages[-max_messages:]

    def _extract_profile_hints(self, profile: UserFinancialProfile, text: str):
        """
        Lightweight regex extractor to update user profile automatically
        when income or deductions are mentioned.
        Examples: 'my salary is 12 lakhs', 'earned 15 LPA', '80C of 1.5 lakh'
        """
        lower = text.lower()

        # Extract income patterns: e.g. "12 lakh", "12,00,000", "15 lpa"
        lakh_income_match = re.search(r'(?:salary|income|package|earning|ctc)\s+(?:is\s+|of\s+)?(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|lpa)', lower)
        if lakh_income_match:
            try:
                val = float(lakh_income_match.group(1)) * 100000
                profile.gross_income = val
            except ValueError:
                pass
        else:
            raw_num_match = re.search(r'(?:salary|income|package)\s+(?:is\s+|of\s+)?(?:rs\.?|inr|₹)?\s*(\d{5,8})', lower)
            if raw_num_match:
                try:
                    profile.gross_income = float(raw_num_match.group(1))
                except ValueError:
                    pass

        # Check regime preference
        if "old regime" in lower and "prefer" in lower or "want old" in lower:
            profile.preferred_regime = "old"
        elif "new regime" in lower and "prefer" in lower or "want new" in lower:
            profile.preferred_regime = "new"

        # 80C hints
        c_match = re.search(r'80c\s*(?:investment|deduction|saving|pf|ppf)?\s*(?:is\s+|of\s+)?(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|k)?', lower)
        if c_match:
            try:
                raw_v = float(c_match.group(1))
                if "lakh" in c_match.group(0):
                    raw_v *= 100000
                elif "k" in c_match.group(0):
                    raw_v *= 1000
                profile.section_80c = min(raw_v, 150000)
            except ValueError:
                pass


# Global singleton
memory_manager = MemoryManager(max_history_turns=6)
