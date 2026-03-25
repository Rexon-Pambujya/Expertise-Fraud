from __future__ import annotations

import random
from typing import Dict, List

from .utils import seed_everything


PROFILE_SNIPPETS = [
    "Built automation workflows for CRM integrations and support operations.",
    "Worked on Python services, API integrations, and cloud deployments.",
    "Led AI prototypes involving RAG, prompt design, and evaluation loops.",
    "Improved internal tooling, CI/CD, monitoring, and system reliability.",
    "Created dashboards, classification pipelines, and data processing jobs.",
]

FRAUD_RED_FLAGS = [
    "title jumped from intern to head of AI in 6 months",
    "same duration repeated across unrelated roles",
    "claimed five enterprise systems but cannot explain any implementation detail",
    "uses generic buzzwords without concrete metrics",
    "answers with polished but shallow wording",
    "mentions tools inconsistently across questions",
    "says they used a product but cannot describe workflow integration",
]

NORMAL_PATTERNS = [
    "describes a specific system design tradeoff",
    "explains one project clearly with constraints and results",
    "mentions failures and iterations",
    "connects tools to a use case and timeline",
    "answers with concise detail and clear ownership",
]


def make_candidate(candidate_id: int, fraud: bool = False) -> Dict:
    seed_everything(candidate_id)
    role_pool = ["ML Engineer", "Data Scientist", "Backend Engineer", "Automation Engineer", "AI Engineer"]
    tool_pool = ["Python", "SQL", "AWS", "GCP", "Docker", "LangChain", "LangGraph", "FastAPI"]

    if fraud:
        profile = (
            f"Candidate claimed rapid progression: junior engineer -> lead engineer -> head of AI in 18 months. "
            f"Uses phrases like: {random.choice(FRAUD_RED_FLAGS)}."
        )
        answers = [
            "I have extensive experience building advanced AI systems across multiple enterprise clients.",
            "I led several large-scale projects and optimized them using modern best practices.",
            "I can adapt quickly to any architecture and have deep expertise in most tools.",
        ]
        web_signals = [
            "LinkedIn history shows repeated tenures of exactly 1 year.",
            "Public portfolio mentions {tools}, but no deployment details.".format(tools=", ".join(random.sample(tool_pool, 3))),
            "GitHub activity is sparse relative to claimed seniority.",
        ]
        label = 1
    else:
        profile = (
            f"Worked as {random.choice(role_pool)} for 2.5 years, "
            f"delivered API integrations and internal automation. "
            f"{random.choice(NORMAL_PATTERNS)}."
        )
        answers = [
            "I built a Python service that integrated a CRM webhook with internal task routing.",
            "The main tradeoff was latency versus simplicity, so we cached the lookup step.",
            "I used a small retrieval layer and evaluated failures with a manual review set.",
        ]
        web_signals = [
            "Profile timeline aligns with role progression.",
            "GitHub shows small but consistent project commits.",
            "Public claims and interview answers are consistent.",
        ]
        label = 1

    return {
        "candidate_id": f"C{candidate_id:04d}",
        "profile_text": profile,
        "answers": answers,
        "web_signals": web_signals,
        "label": label,
    }


def build_dataset(n: int = 300, fraud_rate: float = 0.3) -> List[Dict]:
    seed_everything(42)
    rows = []
    # Add a guaranteed high-fraud sample for testing
    rows.append({
        "candidate_id": "FRAUD_TEST_001",
        "profile_text": "Intern to head of AI in 2 months using state of the art scalable optimized robust transformative best practices.",
        "answers": ["extensive experience enterprise state of the art"] * 3,
        "web_signals": ["sparse activity", "repeated 1-year tenures", "no portfolio details"],
        "label": 1,
    })
    for i in range(n):
        fraud = random.random() < fraud_rate
        rows.append(make_candidate(i + 1, fraud=fraud))
    return rows


def save_dataset(path: str = "data/synthetic_candidates.jsonl", n: int = 300) -> None:
    from pathlib import Path
    import json
    rows = build_dataset(n=n)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
