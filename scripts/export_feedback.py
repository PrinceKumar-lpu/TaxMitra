"""
CLI script to export user downvoted questions from data/feedback.db
into evals/candidates.json for test suite expansion.
"""

import json
from pathlib import Path
from app.services.feedback import get_downvoted_feedback

CANDIDATES_FILE = Path(__file__).resolve().parent.parent / "evals" / "candidates.json"


def export_candidates():
    print("=== Exporting Downvoted Feedback to Evaluation Candidates ===")
    downvoted = get_downvoted_feedback()
    print(f"Found {len(downvoted)} downvoted user interactions.")

    candidates = []
    for idx, row in enumerate(downvoted, start=1):
        candidates.append({
            "id": f"candidate_{row['id']}",
            "category": "user_feedback_candidate",
            "query": row["query"],
            "bot_answer_given": row["bot_answer"],
            "user_comment": row.get("comment"),
            "retrieval_mode_used": row.get("retrieval_mode"),
            "status": "pending_review"
        })

    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"Successfully exported {len(candidates)} candidates to: {CANDIDATES_FILE}")


if __name__ == "__main__":
    export_candidates()
