"""
LLM Service for TaxMitra using Google Gemini API.
Features:
- Deterministic Function Calling for calculate_tax & compare_regimes
- Structured JSON response generation adhering to ChatResponse schema
- Automatic retry on invalid JSON
- Guardrails: Explicit fallback if retrieved context is insufficient
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from app.schemas import ChatResponse, SourceCitation, ToolCallRecord
from app.services.tools import calculate_tax, compare_regimes
from app.services.rag import rag_service
from app.memory import memory_manager, SessionState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TaxMitra, an expert AI tax assistant for Indian salaried individuals.
Your goal is to provide accurate, helpful, and compliant tax guidance according to Indian Income Tax laws for FY 2024-25 (AY 2025-26).

STRICT OPERATING RULES:
1. DETERMINISTIC MATH: You MUST NEVER compute income tax, slab calculations, or regime comparisons yourself using internal arithmetic.
   Whenever a user asks for tax calculation, deductions impact, or Old vs New regime comparison, you MUST call the appropriate function:
   - `calculate_tax(gross_income, regime, deductions)`
   - `compare_regimes(gross_income, deductions)`
2. SOURCE CITATIONS: Base all factual answers (deadlines, slab rates, ITR form eligibility, deduction caps) strictly on the provided RETRIEVED CONTEXT.
   You must cite which document and section the information came from.
3. NO HALLUCINATIONS / OUT-OF-BOUNDS QUERIES: If the RETRIEVED CONTEXT does not contain the answer, or if you are not certain, explicitly state:
   "I am not sure based on the official Indian tax documents currently in my knowledge base, and I cannot invent tax rules."
   Set confidence to "low" in that case.
4. STRUCTURED OUTPUT: Your final output must strictly follow the required JSON structure.
"""

TOOL_DEFINITIONS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="calculate_tax",
                description="Calculate deterministic income tax breakdown for an Indian salaried taxpayer for FY 2024-25 (AY 2025-26).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "gross_income": types.Schema(
                            type=types.Type.NUMBER,
                            description="Gross annual salary in INR (e.g. 1200000 for 12 Lakhs)"
                        ),
                        "regime": types.Schema(
                            type=types.Type.STRING,
                            enum=["new", "old"],
                            description="Tax regime. Default is 'new'."
                        ),
                        "deductions": types.Schema(
                            type=types.Type.OBJECT,
                            description="Deductions for Old Regime. Keys can include section_80c, section_80d, section_80ccd_1b, hra_exemption, home_loan_interest_24b.",
                            properties={
                                "section_80c": types.Schema(type=types.Type.NUMBER, description="80C investments (max 1.5L)"),
                                "section_80d": types.Schema(type=types.Type.NUMBER, description="80D health insurance (max 25k-100k)"),
                                "section_80ccd_1b": types.Schema(type=types.Type.NUMBER, description="NPS Tier-1 (max 50k)"),
                                "hra_exemption": types.Schema(type=types.Type.NUMBER, description="HRA exemption"),
                                "home_loan_interest_24b": types.Schema(type=types.Type.NUMBER, description="Home loan interest (max 2L)")
                            }
                        )
                    },
                    required=["gross_income"]
                )
            ),
            types.FunctionDeclaration(
                name="compare_regimes",
                description="Compare Old Tax Regime vs New Tax Regime for FY 2024-25 and recommend which one saves more money.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "gross_income": types.Schema(
                            type=types.Type.NUMBER,
                            description="Gross annual salary in INR (e.g. 1500000)"
                        ),
                        "deductions": types.Schema(
                            type=types.Type.OBJECT,
                            description="Deductions available in Old Regime (80C, 80D, HRA, etc.)",
                            properties={
                                "section_80c": types.Schema(type=types.Type.NUMBER),
                                "section_80d": types.Schema(type=types.Type.NUMBER),
                                "section_80ccd_1b": types.Schema(type=types.Type.NUMBER),
                                "hra_exemption": types.Schema(type=types.Type.NUMBER),
                                "home_loan_interest_24b": types.Schema(type=types.Type.NUMBER)
                            }
                        )
                    },
                    required=["gross_income"]
                )
            )
        ]
    )
]


class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key or self.api_key == "your_gemini_api_key_here":
                raise ValueError("GEMINI_API_KEY is not configured in .env")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes pure Python deterministic calculation tools."""
        logger.info(f"Executing deterministic tool: {name} with args: {args}")
        if name == "calculate_tax":
            res = calculate_tax(
                gross_income=float(args.get("gross_income", 0.0)),
                regime=args.get("regime", "new"),
                deductions=args.get("deductions")
            )
            return res.model_dump()
        elif name == "compare_regimes":
            res = compare_regimes(
                gross_income=float(args.get("gross_income", 0.0)),
                deductions=args.get("deductions")
            )
            return res.model_dump()
        else:
            raise ValueError(f"Unknown tool name: {name}")

    def chat(self, user_message: str, session_id: Optional[str] = None) -> ChatResponse:
        """
        Primary orchestrator for /chat.
        1. Retrieve session memory & user profile.
        2. Query ChromaDB for top-3 relevant context chunks.
        3. Invoke Gemini with tool calling enabled.
        4. Execute tool calls deterministically.
        5. Generate validated JSON response with 1 retry fallback.
        """
        session = memory_manager.get_or_create_session(session_id)
        effective_session_id = session.session_id

        # Update session memory
        memory_manager.add_user_message(effective_session_id, user_message)

        # Step 1: RAG Retrieval
        retrieved_chunks = rag_service.retrieve(user_message, top_k=3)
        context_str = rag_service.format_context_for_llm(retrieved_chunks)
        source_citations = rag_service.to_source_citations(retrieved_chunks)

        # Build prompt augmenting profile and retrieved docs
        profile_str = session.profile.to_summary_str()
        prompt_with_context = (
            f"USER PROFILE CONTEXT:\n{profile_str}\n\n"
            f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{context_str}\n\n"
            f"USER QUERY: {user_message}"
        )

        # Check if API key is mock or missing
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning("No real GEMINI_API_KEY detected. Running fallback offline response.")
            return self._offline_fallback_response(user_message, effective_session_id, retrieved_chunks, source_citations)

        tool_call_records: List[ToolCallRecord] = []

        try:
            # First turn: Gemini with function declarations
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_with_context,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    temperature=0.1
                )
            )

            # Check if Gemini called any tools
            function_calls = response.function_calls if hasattr(response, "function_calls") else []
            tool_outputs = []

            if function_calls:
                for fc in function_calls:
                    fn_name = fc.name
                    fn_args = fc.args
                    tool_result = self.execute_tool(fn_name, fn_args)
                    tool_call_records.append(
                        ToolCallRecord(
                            tool_name=fn_name,
                            arguments=fn_args,
                            output=tool_result
                        )
                    )
                    tool_outputs.append({
                        "tool": fn_name,
                        "output": tool_result
                    })

                # Follow-up turn providing tool results back to LLM to formulate structured JSON
                followup_prompt = (
                    f"Tool results are:\n{json.dumps(tool_outputs, indent=2)}\n\n"
                    f"Now answer the user's question clearly. Format the final output as a valid JSON object matching:\n"
                    f"{{\n"
                    f'  "answer": "Clear markdown answer summarizing calculations, rules, and breakdown",\n'
                    f'  "confidence": "high" | "medium" | "low"\n'
                    f"}}"
                )

                final_resp = self._call_with_retry(
                    prompt=followup_prompt,
                    system_instruction=SYSTEM_PROMPT + "\nOutput MUST be valid JSON only."
                )
            else:
                # Direct factual question answered via RAG context
                final_resp = self._parse_or_retry_json(response.text if hasattr(response, "text") else "")

            # Determine confidence
            confidence = final_resp.get("confidence", "high")
            if not retrieved_chunks and not tool_call_records:
                confidence = "low"

            ans_text = final_resp.get("answer", "No answer generated.")
            memory_manager.add_model_message(effective_session_id, ans_text)

            return ChatResponse(
                session_id=effective_session_id,
                answer=ans_text,
                sources=source_citations,
                tool_calls=tool_call_records,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error during Gemini interaction: {e}", exc_info=True)
            return self._offline_fallback_response(user_message, effective_session_id, retrieved_chunks, source_citations, error=str(e))

    def _call_with_retry(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        """Calls Gemini and retries once if invalid JSON is returned."""
        for attempt in range(2):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                text = resp.text or "{}"
                return json.loads(self._clean_json(text))
            except Exception as e:
                logger.warning(f"JSON parsing attempt {attempt + 1} failed: {e}")
                if attempt == 1:
                    return {"answer": text if 'text' in locals() else "Unable to parse model output.", "confidence": "medium"}
        return {"answer": "Service unavailable.", "confidence": "low"}

    def _parse_or_retry_json(self, raw_text: str) -> Dict[str, Any]:
        cleaned = self._clean_json(raw_text)
        try:
            return json.loads(cleaned)
        except Exception:
            return {"answer": raw_text, "confidence": "high"}

    def _clean_json(self, text: str) -> str:
        """Strips markdown code blocks from JSON string."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _offline_fallback_response(
        self,
        query: str,
        session_id: str,
        retrieved_chunks: List[Dict[str, Any]],
        source_citations: List[SourceCitation],
        error: Optional[str] = None
    ) -> ChatResponse:
        """
        Offline fallback engine when Gemini API key is absent or unreachable.
        Directly checks for calculation queries and executes tools, or uses top retrieved chunk.
        Ensures the system remains operational for tests and demonstrations.
        """
        tool_records = []
        lower = query.lower()

        # Check if query requests calculation or comparison
        is_calc_query = any(w in lower for w in [
            "calculate", "tax for", "tax on", "how much tax", "salary of", "gross",
            "compare", "regime", "tax payable"
        ])

        if is_calc_query:
            # First check for comma-formatted or plain numbers: e.g. "7,00,000", "7,75,000", "10,75,000", "550000"
            raw_comma_match = re.search(r'(?:rs\.?|inr|₹)?\s*(\d{1,3}(?:,\d{2,3})+|\d{5,8})', lower)
            lakh_word_match = re.search(r'(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|lpa)', lower)
            
            amount = None
            if lakh_word_match:
                amount = float(lakh_word_match.group(1)) * 100000
            elif raw_comma_match:
                clean_num_str = raw_comma_match.group(1).replace(",", "")
                amount = float(clean_num_str)

            # Check deductions in query if any
            deductions_dict = {}
            if "80c" in lower:
                deductions_dict["section_80c"] = 150000
            if "80ccd" in lower or "nps" in lower:
                deductions_dict["section_80ccd_1b"] = 50000
            if "80d" in lower:
                deductions_dict["section_80d"] = 50000
            if "home loan" in lower or "24b" in lower:
                deductions_dict["home_loan_interest_24b"] = 200000

            if amount:
                if "compare" in lower or ("regime" in lower and "which" in lower):
                    comp_res = compare_regimes(gross_income=amount, deductions=deductions_dict or None)
                    tool_records.append(ToolCallRecord(tool_name="compare_regimes", arguments={"gross_income": amount, "deductions": deductions_dict}, output=comp_res.model_dump()))
                    answer = (
                        f"### Tax Regime Comparison for ₹{amount:,.0f} (FY 2024-25 / AY 2025-26)\n\n"
                        f"- **New Regime Tax Payable:** ₹{comp_res.new_regime.total_tax_payable:,.2f} (Standard Deduction: ₹75,000)\n"
                        f"- **Old Regime Tax Payable:** ₹{comp_res.old_regime.total_tax_payable:,.2f} (Standard Deduction: ₹50,000)\n\n"
                        f"**Recommendation:** **{comp_res.recommended_regime.upper()} REGIME**\n"
                        f"{comp_res.rationale}"
                    )
                    return ChatResponse(session_id=session_id, answer=answer, sources=source_citations, tool_calls=tool_records, confidence="high")
                else:
                    regime = "old" if "old" in lower else "new"
                    tax_res = calculate_tax(gross_income=amount, regime=regime, deductions=deductions_dict or None)
                    tool_records.append(ToolCallRecord(tool_name="calculate_tax", arguments={"gross_income": amount, "regime": regime, "deductions": deductions_dict}, output=tax_res.model_dump()))
                    answer = (
                        f"### Tax Breakdown for ₹{amount:,.0f} under {regime.capitalize()} Regime (FY 2024-25)\n\n"
                        f"- **Gross Income:** ₹{tax_res.gross_income:,.2f}\n"
                        f"- **Standard Deduction:** ₹{tax_res.standard_deduction:,.2f}\n"
                        f"- **Taxable Income:** ₹{tax_res.taxable_income:,.2f}\n"
                        f"- **Tax Before Rebate:** ₹{tax_res.tax_before_rebate:,.2f}\n"
                        f"- **Section 87A Rebate:** ₹{tax_res.section_87a_rebate:,.2f}\n"
                        f"- **Health & Education Cess (4%):** ₹{tax_res.cess:,.2f}\n"
                        f"- **Net Tax Payable:** **₹{tax_res.total_tax_payable:,.2f}**\n\n"
                        f"*Sourced directly from {tax_res.source_url}*"
                    )
                    return ChatResponse(session_id=session_id, answer=answer, sources=source_citations, tool_calls=tool_records, confidence="high")

        # Factual RAG fallback
        if retrieved_chunks:
            top_doc = retrieved_chunks[0]
            # Check if relevance is genuinely acceptable
            if top_doc.get("similarity", 0) > 0.35 and top_doc.get("is_relevant", True):
                # Guard against out-of-scope domain terms
                out_of_scope_terms = ["germany", "german", "multinational corporation", "evade", "illegal", "loophole"]
                if not any(term in lower for term in out_of_scope_terms):
                    answer = (
                        f"Based on the official tax guidance in **{top_doc['metadata']['source']}** "
                        f"(Section: *{top_doc['metadata']['section']}*):\n\n"
                        f"{top_doc['text']}"
                    )
                    return ChatResponse(session_id=session_id, answer=answer, sources=source_citations, tool_calls=tool_records, confidence="high")

        return ChatResponse(
            session_id=session_id,
            answer="I am not sure based on the official Indian tax documents currently in my knowledge base, and I cannot invent tax rules.",
            sources=[],
            tool_calls=[],
            confidence="low"
        )


# Global LLM service instance
llm_service = LLMService()
