from __future__ import annotations

import argparse
import json

from .synthetic_data import build_dataset
from .graph import app as graph_app
from .feature_engineering import build_feature_report, risk_from_feature_report
from .agent import choose_action_from_report


def demo(candidate_index: int = 0):
    data = build_dataset(n=50)
    candidate = data[candidate_index % len(data)]
    fr = build_feature_report(candidate["profile_text"], candidate["answers"], candidate["web_signals"])
    out = {
        "candidate_id": candidate["candidate_id"],
        "feature_report": fr.model_dump(),
        "risk_score": risk_from_feature_report(fr),
        "decision": choose_action_from_report(fr.model_dump()),
        "graph": graph_app.invoke({
            "candidate_id": candidate["candidate_id"],
            "profile_text": candidate["profile_text"],
            "answers": candidate["answers"],
            "web_signals": candidate["web_signals"],
        }),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, default=0)
    args = parser.parse_args()
    demo(args.candidate)
