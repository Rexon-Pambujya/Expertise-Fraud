from __future__ import annotations

import re
from typing import List, Tuple

from .schemas import FeatureReport
from .utils import avg, soft_clip, summarize_tokens, flatten_evidence


BUZZWORDS = [
    "state of the art", "enterprise", "scalable", "optimized",
    "robust", "transformative", "best practices"
]

TOOL_NAMES = ["python", "sql", "aws", "gcp", "docker", "langchain", "langgraph", "fastapi", "pytorch"]


def _contains_any(text: str, keywords: List[str]) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


def compute_timeline_anomaly(profile_text: str) -> Tuple[float, List[str]]:
    txt = profile_text.lower()
    evidence = []
    score = 0.0

    if re.search(r"intern\s*(?:->|to)\s*(?:lead|head)|junior\s*(?:->|to)\s*(?:lead|head)", txt):
        score += 0.45
        evidence.append("implausible title jump in a short period")

    if re.search(r"\b(6 months|3 months|1 month|18 months|2 months)\b", txt):
        score += 0.25
        evidence.append("compressed progression timeline")

    if "claimed rapid progression" in txt:
        score += 0.2
        evidence.append("self-described rapid progression")

    if _contains_any(txt, ["head of ai", "principal architect", "vp"]) and "years" not in txt:
        score += 0.2
        evidence.append("senior title without supporting tenure detail")

    return soft_clip(score), evidence


def compute_answer_anomaly(answers: List[str]) -> Tuple[float, List[str]]:
    joined = " ".join(answers)
    low = joined.lower()
    evidence = []
    score = 0.0

    token_counts = [summarize_tokens(a) for a in answers]
    if token_counts and avg(token_counts) > 22:
        score += 0.1
        evidence.append("answers are verbose and generic")

    buzz_count = sum(1 for b in BUZZWORDS if b in low)
    if buzz_count >= 3:
        score += 0.25
        evidence.append("heavy buzzword usage")

    if any(len(a.split()) < 8 for a in answers):
        score += 0.05
        evidence.append("one answer is unusually thin")

    if "extensive experience" in low and not any(c in low for c in ["built", "integrated", "cached", "evaluated", "tradeoff"]):
        score += 0.2
        evidence.append("claims expertise without implementation detail")

    return soft_clip(score), evidence


def compute_consistency_anomaly(profile_text: str, answers: List[str]) -> Tuple[float, List[str]]:
    text = (profile_text + " " + " ".join(answers)).lower()
    evidence = []
    score = 0.0

    tool_hits = [tool for tool in TOOL_NAMES if tool in text]
    if len(set(tool_hits)) >= 5 and "project" not in text:
        score += 0.15
        evidence.append("broad tool claims without anchoring project context")

    if "used a product" in text and "workflow" not in text and "integration" not in text:
        score += 0.15
        evidence.append("product claim lacks workflow description")

    if "multiple enterprise clients" in text and "specific" not in text and "one" not in text:
        score += 0.1
        evidence.append("claims scale but no concrete example")

    repeated_gas = sum(text.count(word) for word in ["scalable", "robust", "optimized", "enterprise"])
    if repeated_gas >= 4:
        score += 0.1
        evidence.append("repeated generic phrasing across answers")

    return soft_clip(score), evidence


def compute_web_anomaly(web_signals: List[str]) -> Tuple[float, List[str]]:
    joined = " ".join(web_signals).lower()
    evidence = []
    score = 0.0

    if "repeated tenures" in joined:
        score += 0.2
        evidence.append("profile timeline repetition on public signals")

    if "sparse" in joined and "claimed seniority" in joined:
        score += 0.2
        evidence.append("weak public footprint relative to claimed seniority")

    if "aligns" in joined and "consistent" in joined:
        score -= 0.05

    return soft_clip(score), evidence


def build_feature_report(profile_text: str, answers: List[str], web_signals: List[str]) -> FeatureReport:
    timeline_score, timeline_ev = compute_timeline_anomaly(profile_text)
    answer_score, answer_ev = compute_answer_anomaly(answers)
    consistency_score, consistency_ev = compute_consistency_anomaly(profile_text, answers)
    web_score, web_ev = compute_web_anomaly(web_signals)

    evidence = flatten_evidence(timeline_ev + answer_ev + consistency_ev + web_ev)
    summary = " | ".join(evidence[:4]) if evidence else "No major red flags detected."

    return FeatureReport(
        timeline_anomaly=timeline_score,
        answer_anomaly=answer_score,
        consistency_anomaly=consistency_score,
        web_anomaly=web_score,
        evidence=evidence,
        summary=summary,
    )


def risk_from_feature_report(report: FeatureReport) -> float:
    weighted = (
        0.35 * report.timeline_anomaly
        + 0.30 * report.answer_anomaly
        + 0.20 * report.consistency_anomaly
        + 0.15 * report.web_anomaly
    )
    return soft_clip(weighted)
