"""
Test suite for TaxMitra API endpoints, RAG retrieval, and conversation memory.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.memory import memory_manager
from app.services.rag import rag_service

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "TaxMitra"
    assert data["financial_year"] == "2024-2025"


def test_calculate_endpoint_new_regime():
    payload = {
        "gross_income": 775000,
        "regime": "new"
    }
    response = client.post("/calculate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["gross_income"] == 775000
    assert data["standard_deduction"] == 75000
    assert data["total_tax_payable"] == 0.0


def test_compare_endpoint():
    payload = {
        "gross_income": 1200000,
        "deductions": {
            "section_80c": 150000,
            "section_80d": 25000
        }
    }
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_regime" in data
    assert "old_regime" in data
    assert "new_regime" in data


def test_chat_endpoint_calculation():
    payload = {
        "message": "Calculate my tax on 7.75 lakhs under new regime"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "session_id" in data
    assert data["confidence"] in ["high", "medium", "low"]


def test_chat_memory_profile_extraction():
    session_id = "test-session-123"
    payload1 = {
        "message": "My gross salary is 15 lakhs and I prefer new regime",
        "session_id": session_id
    }
    client.post("/chat", json=payload1)

    session = memory_manager.get_or_create_session(session_id)
    assert session.profile.gross_income == 1500000
    assert session.profile.preferred_regime == "new"


def test_rate_limiter_does_not_block_normal_traffic():
    for _ in range(5):
        resp = client.get("/health")
        assert resp.status_code == 200
