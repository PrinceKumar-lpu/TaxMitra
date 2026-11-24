"""
Unit tests for deterministic tax tools (calculate_tax & compare_regimes).
Verifies:
- Standard deduction (₹75k for new, ₹50k for old)
- Section 87A rebate boundary conditions (₹7L for new, ₹5L for old)
- Higher income slabs
- 4% Health & Education cess
- Old vs New regime comparison logic
"""

import pytest
from app.services.tools import calculate_tax, compare_regimes, DeductionsInput


def test_new_regime_zero_tax_up_to_7_75_lakh():
    """
    In FY 2024-25 New Regime:
    Gross = 7,75,000
    Standard Deduction = 75,000
    Taxable Income = 7,00,000
    Tax before rebate:
      0 to 3L: 0
      3L to 7L: 4L * 5% = 20,000
    Rebate under Section 87A: 20,000 (taxable income <= 7L)
    Net Tax = 0
    """
    res = calculate_tax(gross_income=775000, regime="new")
    assert res.standard_deduction == 75000
    assert res.taxable_income == 700000
    assert res.tax_before_rebate == 20000
    assert res.section_87a_rebate == 20000
    assert res.tax_after_rebate == 0
    assert res.total_tax_payable == 0


def test_new_regime_taxable_above_7_lakh():
    """
    Gross = 10,75,000
    Standard Deduction = 75,000
    Taxable Income = 10,00,000
    Slabs:
      0 - 3L: 0
      3L - 7L: 4,00,000 * 5% = 20,000
      7L - 10L: 3,00,000 * 10% = 30,000
    Tax before rebate = 50,000
    Rebate 87A: 0 (taxable income > 7L)
    Cess 4%: 2,000
    Total Tax = 52,000
    """
    res = calculate_tax(gross_income=1075000, regime="new")
    assert res.standard_deduction == 75000
    assert res.taxable_income == 1000000
    assert res.tax_before_rebate == 50000
    assert res.section_87a_rebate == 0
    assert res.cess == 2000
    assert res.total_tax_payable == 52000


def test_old_regime_zero_tax_up_to_5_5_lakh():
    """
    In Old Regime:
    Gross = 5,50,000
    Standard Deduction = 50,000
    Taxable Income = 5,00,000
    Slabs:
      0 - 2.5L: 0
      2.5L - 5L: 2,50,000 * 5% = 12,500
    Rebate 87A: 12,500 (taxable income <= 5L)
    Total Tax = 0
    """
    res = calculate_tax(gross_income=550000, regime="old")
    assert res.standard_deduction == 50000
    assert res.taxable_income == 500000
    assert res.tax_before_rebate == 12500
    assert res.section_87a_rebate == 12500
    assert res.total_tax_payable == 0


def test_old_regime_with_deductions_80c_and_80d():
    """
    Gross = 10,00,000
    Standard deduction = 50,000
    80C = 1,50,000 (claimed 2,00,000, capped at 1,50,000)
    80D = 25,000
    Total Deductions = 1,75,000
    Taxable Income = 10,00,000 - 50,000 - 1,75,000 = 7,75,000
    Slabs:
      0 - 2.5L: 0
      2.5L - 5L: 2,50,000 * 5% = 12,500
      5L - 7.75L: 2,75,000 * 20% = 55,000
    Tax before rebate = 67,500
    Cess 4%: 2,700
    Total Tax = 70,200
    """
    deductions = {
        "section_80c": 200000,  # exceeds cap
        "section_80d": 25000
    }
    res = calculate_tax(gross_income=1000000, regime="old", deductions=deductions)
    assert res.eligible_deductions == 175000
    assert res.taxable_income == 775000
    assert res.tax_before_rebate == 67500
    assert res.cess == 2700
    assert res.total_tax_payable == 70200


def test_compare_regimes_recommends_old_when_deductions_are_high():
    """
    When salaried person has ₹4L+ in valid deductions (80C, HRA, 80D, 24b),
    Old Regime can save more tax.
    """
    deductions = {
        "section_80c": 150000,
        "section_80d": 25000,
        "section_80ccd_1b": 50000,
        "hra_exemption": 180000
    }
    comp = compare_regimes(gross_income=1200000, deductions=deductions)
    assert comp.recommended_regime == "old"
    assert comp.savings > 0
    assert comp.old_regime.total_tax_payable < comp.new_regime.total_tax_payable


def test_compare_regimes_recommends_new_when_no_deductions():
    """
    With zero deductions at ₹12L gross salary, New Regime should win by a substantial margin.
    """
    comp = compare_regimes(gross_income=1200000, deductions=None)
    assert comp.recommended_regime == "new"
    assert comp.savings > 0
    assert comp.new_regime.total_tax_payable < comp.old_regime.total_tax_payable


def test_invalid_regime_raises_value_error():
    with pytest.raises(ValueError, match="Invalid regime"):
        calculate_tax(gross_income=1000000, regime="invalid_regime")
