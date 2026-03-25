from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class FeatureReport(BaseModel):
    timeline_anomaly: float = Field(ge=0.0, le=1.0, description="Suspicious profile history changes")
    answer_anomaly: float = Field(ge=0.0, le=1.0, description="Suspicious screening answer characteristics")
    consistency_anomaly: float = Field(ge=0.0, le=1.0, description="Mismatch across answers or claims")
    web_anomaly: float = Field(ge=0.0, le=1.0, description="Inconsistency against public web signals")
    evidence: List[str] = Field(default_factory=list)
    summary: str = ""


class CandidateState(BaseModel):
    candidate_id: str
    profile_text: str
    answers: List[str] = Field(default_factory=list)
    web_signals: List[str] = Field(default_factory=list)
    feature_report: Optional[FeatureReport] = None
    risk_score: float = 0.0
    decision: Optional[Literal["PASS", "FLAG", "ASK_MORE"]] = None
    step: int = 0
    history: List[str] = Field(default_factory=list)


class DecisionRecord(BaseModel):
    candidate_id: str
    risk_score: float
    decision: str
    rationale: List[str]
