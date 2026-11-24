"""
Deterministic tax calculation service for TaxMitra.
Interview Context:
- Why separate this from LLM? LLMs suffer from floating-point hallucination and non-deterministic arithmetic.
  By isolating tax math in a tested, pure Python module driven by data/tax_rules.json, we guarantee 100% calculation
  correctness and easy updates when the Finance Bill changes slabs or rebate rules.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

RULES_FILE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "tax_rules.json"


def load_tax_rules() -> Dict[str, Any]:
    """Loads externalized tax rules from data/tax_rules.json."""
    if not RULES_FILE_PATH.exists():
        raise FileNotFoundError(f"Tax rules file not found at {RULES_FILE_PATH}")
    with open(RULES_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class DeductionsInput(BaseModel):
    """Optional user deductions for Old Regime."""
    section_80c: float = Field(default=0.0, ge=0.0, description="Deduction under Section 80C (max 1.5L)")
    section_80d: float = Field(default=0.0, ge=0.0, description="Deduction under Section 80D (health insurance)")
    section_80ccd_1b: float = Field(default=0.0, ge=0.0, description="Additional NPS deduction (max 50k)")
    hra_exemption: float = Field(default=0.0, ge=0.0, description="House Rent Allowance exemption")
    home_loan_interest_24b: float = Field(default=0.0, ge=0.0, description="Home loan interest Section 24(b)")
    other_deductions: float = Field(default=0.0, ge=0.0, description="Other allowable exemptions/deductions")


class SlabBreakdown(BaseModel):
    slab: str
    taxable_in_slab: float
    rate_percent: float
    tax_amount: float


class TaxCalculationResult(BaseModel):
    financial_year: str
    assessment_year: str
    regime: str
    gross_income: float
    standard_deduction: float
    eligible_deductions: float
    taxable_income: float
    slab_breakdown: List[SlabBreakdown]
    tax_before_rebate: float
    section_87a_rebate: float
    tax_after_rebate: float
    cess: float
    total_tax_payable: float
    source_url: str


class RegimeComparisonResult(BaseModel):
    gross_income: float
    old_regime: TaxCalculationResult
    new_regime: TaxCalculationResult
    tax_difference: float
    recommended_regime: str
    savings: float
    rationale: str


def calculate_tax(
    gross_income: float,
    regime: str = "new",
    deductions: Optional[Dict[str, float] | DeductionsInput] = None
) -> TaxCalculationResult:
    """
    Calculates income tax deterministically for Indian salaried individuals.
    
    Args:
        gross_income: Total annual gross salary (INR).
        regime: 'new' or 'old'. Default is 'new'.
        deductions: Dict or DeductionsInput with 80C, 80D, HRA, etc. (applicable primarily to Old Regime).
    """
    rules = load_tax_rules()
    regime_normalized = regime.strip().lower()
    if regime_normalized not in ["old", "new"]:
        raise ValueError(f"Invalid regime '{regime}'. Must be 'old' or 'new'.")

    # Standard deduction
    std_ded = rules["standard_deduction"].get(f"{regime_normalized}_regime", 0.0)

    # Process deductions
    if isinstance(deductions, dict):
        ded_obj = DeductionsInput(**deductions)
    elif isinstance(deductions, DeductionsInput):
        ded_obj = deductions
    else:
        ded_obj = DeductionsInput()

    eligible_deductions = 0.0
    if regime_normalized == "old":
        # Apply statutory caps
        c_cap = rules["deduction_limits"]["section_80c"]
        nps_cap = rules["deduction_limits"]["section_80ccd_1b_nps"]
        home_loan_cap = rules["deduction_limits"]["section_24b_home_loan_interest"]

        allowed_80c = min(ded_obj.section_80c, c_cap)
        allowed_nps = min(ded_obj.section_80ccd_1b, nps_cap)
        allowed_home_loan = min(ded_obj.home_loan_interest_24b, home_loan_cap)
        # Note: 80D cap depends on age/parents, capped cleanly at declared or default max
        allowed_80d = ded_obj.section_80d
        allowed_hra = ded_obj.hra_exemption
        allowed_others = ded_obj.other_deductions

        eligible_deductions = (
            allowed_80c + allowed_nps + allowed_home_loan + allowed_80d + allowed_hra + allowed_others
        )
    else:
        # In New Regime, chapter VI-A deductions (80C, 80D, HRA, 24b) are generally not available for salaried
        # Standard deduction is granted automatically
        eligible_deductions = 0.0

    # Calculate taxable income
    taxable_income = max(0.0, gross_income - std_ded - eligible_deductions)

    # Calculate slab tax
    slabs = rules["slabs"][f"{regime_normalized}_regime"]
    slab_breakdowns: List[SlabBreakdown] = []
    tax_before_rebate = 0.0

    for slab in slabs:
        slab_min = slab["min"]
        slab_max = slab["max"]
        rate = slab["rate"]

        if taxable_income > slab_min:
            if slab_max is not None:
                taxable_in_slab = min(taxable_income, slab_max) - slab_min
            else:
                taxable_in_slab = taxable_income - slab_min

            tax_in_slab = taxable_in_slab * rate
            tax_before_rebate += tax_in_slab

            slab_breakdowns.append(
                SlabBreakdown(
                    slab=slab["description"],
                    taxable_in_slab=round(taxable_in_slab, 2),
                    rate_percent=rate * 100,
                    tax_amount=round(tax_in_slab, 2)
                )
            )

    # Section 87A Rebate
    rebate_info = rules["section_87a_rebate"][f"{regime_normalized}_regime"]
    rebate_threshold = rebate_info["threshold_taxable_income"]
    max_rebate = rebate_info["max_rebate"]

    rebate_87a = 0.0
    if taxable_income <= rebate_threshold:
        rebate_87a = min(tax_before_rebate, max_rebate)

    tax_after_rebate = max(0.0, tax_before_rebate - rebate_87a)

    # Health and Education Cess (4%)
    cess_rate = rules.get("cess_rate", 0.04)
    cess = round(tax_after_rebate * cess_rate, 2)
    total_tax = round(tax_after_rebate + cess, 2)

    return TaxCalculationResult(
        financial_year=rules["financial_year"],
        assessment_year=rules["assessment_year"],
        regime=regime_normalized,
        gross_income=round(gross_income, 2),
        standard_deduction=round(std_ded, 2),
        eligible_deductions=round(eligible_deductions, 2),
        taxable_income=round(taxable_income, 2),
        slab_breakdown=slab_breakdowns,
        tax_before_rebate=round(tax_before_rebate, 2),
        section_87a_rebate=round(rebate_87a, 2),
        tax_after_rebate=round(tax_after_rebate, 2),
        cess=cess,
        total_tax_payable=total_tax,
        source_url=rules["source_url"]
    )


def compare_regimes(
    gross_income: float,
    deductions: Optional[Dict[str, float] | DeductionsInput] = None
) -> RegimeComparisonResult:
    """
    Compares Old vs New Regime deterministically and provides an objective recommendation.
    """
    old_res = calculate_tax(gross_income=gross_income, regime="old", deductions=deductions)
    new_res = calculate_tax(gross_income=gross_income, regime="new", deductions=deductions)

    diff = round(abs(old_res.total_tax_payable - new_res.total_tax_payable), 2)

    if new_res.total_tax_payable < old_res.total_tax_payable:
        recommended = "new"
        savings = diff
        rationale = (
            f"The New Tax Regime saves you ₹{savings:,.2f} in tax. "
            f"New Regime tax is ₹{new_res.total_tax_payable:,.2f} vs ₹{old_res.total_tax_payable:,.2f} in Old Regime. "
            f"Under the New Regime you benefit from lower slab rates and a higher Standard Deduction (₹75,000)."
        )
    elif old_res.total_tax_payable < new_res.total_tax_payable:
        recommended = "old"
        savings = diff
        rationale = (
            f"The Old Tax Regime saves you ₹{savings:,.2f} in tax. "
            f"Old Regime tax is ₹{old_res.total_tax_payable:,.2f} vs ₹{new_res.total_tax_payable:,.2f} in New Regime. "
            f"Your deductions of ₹{old_res.eligible_deductions:,.2f} (80C, 80D, HRA, etc.) sufficiently lower your taxable income."
        )
    else:
        recommended = "either"
        savings = 0.0
        rationale = (
            f"Both regimes result in the exact same tax of ₹{new_res.total_tax_payable:,.2f}. "
            f"New Regime is typically preferred due to zero documentation requirement."
        )

    return RegimeComparisonResult(
        gross_income=round(gross_income, 2),
        old_regime=old_res,
        new_regime=new_res,
        tax_difference=diff,
        recommended_regime=recommended,
        savings=savings,
        rationale=rationale
    )
