from __future__ import annotations

import streamlit as st
import pandas as pd

from app.synthetic_data import build_dataset
from app.graph import app as graph_app
from app.feature_engineering import build_feature_report, risk_from_feature_report
from app.agent import choose_action_from_report


st.set_page_config(page_title="Expertise Fraud Detection", layout="wide")

st.title("Expertise Fraud Detection System")
st.caption("LangChain + LangGraph + synthetic RL demo")

if "dataset" not in st.session_state:
    st.session_state.dataset = build_dataset(n=100)

data = st.session_state.dataset
idx = st.slider("Candidate", 0, len(data) - 1, 0)
candidate = data[idx]

col1, col2 = st.columns([1.1, 1.2])

with col1:
    st.subheader("Candidate Snapshot")
    st.write("**Candidate ID:**", candidate["candidate_id"])
    st.write("**Profile Text:**")
    st.info(candidate["profile_text"])
    st.write("**Answers:**")
    for i, ans in enumerate(candidate["answers"], 1):
        st.write(f"{i}. {ans}")
    st.write("**Web Signals:**")
    for sig in candidate["web_signals"]:
        st.write(f"- {sig}")

fr = build_feature_report(candidate["profile_text"], candidate["answers"], candidate["web_signals"])
risk = risk_from_feature_report(fr)
decision = choose_action_from_report(fr.model_dump())

with col2:
    st.subheader("Extracted Red Flags")
    feature_df = pd.DataFrame([{
        "timeline_anomaly": fr.timeline_anomaly,
        "answer_anomaly": fr.answer_anomaly,
        "consistency_anomaly": fr.consistency_anomaly,
        "web_anomaly": fr.web_anomaly,
        "risk_score": risk,
        "decision": decision,
    }])
    st.dataframe(feature_df, use_container_width=True)
    st.metric("Risk score", f"{risk:.2f}")
    st.metric("Decision", decision)
    st.write("**Evidence:**")
    for item in fr.evidence:
        st.write(f"- {item}")

    st.write("**Rationale Summary:**")
    st.success(fr.summary)

st.divider()
st.subheader("LangGraph Run")
graph_state = graph_app.invoke({
    "candidate_id": candidate["candidate_id"],
    "profile_text": candidate["profile_text"],
    "answers": candidate["answers"],
    "web_signals": candidate["web_signals"],
})

st.json(graph_state)

st.divider()
st.subheader("How to present this")
st.write(
    "Use the UI to show that the system is not guessing from one feature. "
    "It combines profile history, answer quality, and public consistency, then routes the case through a decision policy."
)
