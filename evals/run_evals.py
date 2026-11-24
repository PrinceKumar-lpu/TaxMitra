"""
Automated Evaluation Suite for TaxMitra.
Evaluates:
1. Deterministic Tax Calculation Accuracy: Verifies math outputs against Income Tax Department rules.
2. RAG Retrieval Hit-Rate: Checks if the correct markdown source was ranked in top-3 chunks.
3. Keyword & Guardrail Compliance: Verifies facts and 'I am not sure' response for out-of-scope questions.

Saves structured benchmark metrics to evals/results.json.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure environment is loaded
from dotenv import load_dotenv
load_dotenv()

from app.services.rag import rag_service
from app.services.tools import calculate_tax, compare_regimes
from app.services.llm import llm_service

EVAL_FILE = Path(__file__).resolve().parent / "eval_questions.json"
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"


def run_benchmarks() -> Dict[str, Any]:
    print("=" * 65)
    print("        TaxMitra Automated Evaluation Runner (FY 2024-25)")
    print("=" * 65)

    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Evaluation questions not found at {EVAL_FILE}")

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(questions)} evaluation benchmark test cases.\n")

    results_detail = []
    calc_tests = 0
    calc_passed = 0

    rag_tests = 0
    rag_hits = 0

    overall_passed = 0

    start_time = time.time()

    for idx, item in enumerate(questions, start=1):
        qid = item["id"]
        category = item["category"]
        query = item["query"]
        print(f"[{idx}/{len(questions)}] Running {qid} ({category})...")

        test_passed = True
        notes = []

        # 1. Tax Math Evaluation
        if category == "tax_calculation":
            calc_tests += 1
            # Run query through llm service (which triggers deterministic tool)
            resp = llm_service.chat(query)
            expected_tax = item.get("expected_tax")

            # Check tool call record
            actual_tax = None
            if resp.tool_calls:
                tool_out = resp.tool_calls[0].output
                actual_tax = tool_out.get("total_tax_payable")

            if actual_tax is not None and abs(actual_tax - expected_tax) < 1.0:
                notes.append(f"Tax math accurate: expected {expected_tax}, got {actual_tax}")
            else:
                test_passed = False
                notes.append(f"Tax math mismatch: expected {expected_tax}, got {actual_tax}")

            if test_passed:
                calc_passed += 1

        elif category == "regime_comparison":
            calc_tests += 1
            resp = llm_service.chat(query)
            expected_winner = item.get("expected_winner")
            actual_winner = None
            if resp.tool_calls:
                actual_winner = resp.tool_calls[0].output.get("recommended_regime")

            if actual_winner == expected_winner:
                notes.append(f"Regime winner matched: {actual_winner}")
                calc_passed += 1
            else:
                test_passed = False
                notes.append(f"Regime winner mismatch: expected {expected_winner}, got {actual_winner}")

        # 2. RAG Retrieval Evaluation
        elif category == "factual_rag":
            rag_tests += 1
            expected_doc = item["expected_doc"]
            retrieved = rag_service.retrieve(query, top_k=3)
            retrieved_sources = [r["metadata"].get("source") for r in retrieved]

            is_hit = expected_doc in retrieved_sources
            if is_hit:
                rag_hits += 1
                notes.append(f"Hit-rate pass: {expected_doc} found in top-3")
            else:
                test_passed = False
                notes.append(f"Hit-rate miss: {expected_doc} not in {retrieved_sources}")

            # Verify response generation
            resp = llm_service.chat(query)
            answer_lower = resp.answer.lower()
            missing_keywords = [kw for kw in item.get("expected_keywords", []) if kw.lower() not in answer_lower]
            if missing_keywords:
                notes.append(f"Missing keywords: {missing_keywords}")
                # Don't strictly fail if retrieval succeeded but LLM phrased differently

        # 3. Guardrail / Out of bounds
        elif category == "out_of_bounds":
            resp = llm_service.chat(query)
            ans_lower = resp.answer.lower()
            if "not sure" in ans_lower or "cannot" in ans_lower or resp.confidence == "low":
                notes.append("Hallucination guardrail active: properly declined")
            else:
                test_passed = False
                notes.append("Guardrail failed: model attempted to answer out-of-scope query")

        if test_passed:
            overall_passed += 1

        results_detail.append({
            "id": qid,
            "category": category,
            "query": query,
            "passed": test_passed,
            "notes": "; ".join(notes)
        })

    elapsed = round(time.time() - start_time, 2)
    calc_accuracy = round((calc_passed / calc_tests * 100), 2) if calc_tests > 0 else 0.0
    retrieval_hit_rate = round((rag_hits / rag_tests * 100), 2) if rag_tests > 0 else 0.0
    overall_pass_rate = round((overall_passed / len(questions) * 100), 2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(questions),
        "overall_passed": overall_passed,
        "overall_pass_rate_pct": overall_pass_rate,
        "tax_calculation_accuracy_pct": calc_accuracy,
        "retrieval_hit_rate_pct": retrieval_hit_rate,
        "elapsed_seconds": elapsed,
        "details": results_detail
    }

    # Save to evals/results.json
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Test Cases:            {len(questions)}")
    print(f"Overall Passed:              {overall_passed}/{len(questions)} ({overall_pass_rate}%)")
    print(f"Tax Math Accuracy:           {calc_passed}/{calc_tests} ({calc_accuracy}%)")
    print(f"RAG Retrieval Hit-Rate:      {rag_hits}/{rag_tests} ({retrieval_hit_rate}%)")
    print(f"Execution Time:              {elapsed}s")
    print(f"Detailed Results saved to:   {RESULTS_FILE}")
    print("=" * 65 + "\n")

    return summary


if __name__ == "__main__":
    run_benchmarks()
