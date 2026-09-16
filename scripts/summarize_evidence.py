#!/usr/bin/env python3
"""Create an auditable summary of offline relevance and serving latency evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def relative_pct(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0) if baseline else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="data/reports")
    parser.add_argument("--output", default="data/reports/production_evidence.json")
    args = parser.parse_args()
    reports = Path(args.reports)
    hybrid = json.loads((reports / "hybrid_candidates.json").read_text())
    bpr = json.loads((reports / "bpr_ranker.json").read_text())
    latency_path = reports / "latency_benchmark.json"
    latency = json.loads(latency_path.read_text()) if latency_path.exists() else None
    test_candidates = hybrid["test"]["metrics"]
    bpr_test = bpr["test_evaluation"]["metrics"]
    evidence = {
        "candidate_relevance": {
            "cohort": "untouched chronological test warm users",
            "users": hybrid["test"]["eligible_warm_users"],
            "hybrid_recall_at_100": test_candidates["hybrid"]["@100"]["recall"],
            "cooccurrence_recall_at_100": test_candidates["item_cooccurrence"]["@100"]["recall"],
            "relative_recall_at_100_lift_pct": relative_pct(
                test_candidates["hybrid"]["@100"]["recall"],
                test_candidates["item_cooccurrence"]["@100"]["recall"],
            ),
        },
        "neural_ranker": {
            "cohort": "retrieved-positive test candidate groups",
            "groups": bpr["test_evaluation"]["evaluated_groups"],
            "promoted": bpr["candidate_promoted"],
            "neural_ndcg_at_10": bpr_test["neural"]["@10"]["ndcg"],
            "retrieval_ndcg_at_10": bpr_test["cooccurrence"]["@10"]["ndcg"],
            "relative_ndcg_at_10_lift_pct": relative_pct(
                bpr_test["neural"]["@10"]["ndcg"],
                bpr_test["cooccurrence"]["@10"]["ndcg"],
            ),
            "conclusion": "challenger retained in shadow mode; promotion gate not met",
        },
        "latency": latency,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
