from __future__ import annotations

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from .feature_engineering import build_feature_report, risk_from_feature_report
from .agent import choose_action_from_report


class CandidateGraphState(TypedDict, total=False):
    candidate_id: str
    profile_text: str
    answers: List[str]
    web_signals: List[str]
    feature_report: Dict[str, Any]
    risk_score: float
    decision: str
    rationale: List[str]


def extract_features(state: CandidateGraphState) -> CandidateGraphState:
    fr = build_feature_report(state["profile_text"], state.get("answers", []), state.get("web_signals", []))
    state["feature_report"] = fr.model_dump()
    state["risk_score"] = risk_from_feature_report(fr)
    state["rationale"] = fr.evidence[:5]
    return state


def decide(state: CandidateGraphState) -> CandidateGraphState:
    report = state["feature_report"]
    state["decision"] = choose_action_from_report(report)
    return state


def build_graph():
    g = StateGraph(CandidateGraphState)
    g.add_node("extract_features", extract_features)
    g.add_node("decide", decide)
    g.set_entry_point("extract_features")
    g.add_edge("extract_features", "decide")
    g.add_edge("decide", END)
    return g.compile()


app = build_graph()
