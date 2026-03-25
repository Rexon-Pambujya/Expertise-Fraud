from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


FEATURE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict evaluator for expertise fraud. Return concise structured findings only."
    ),
    (
        "human",
        """
Profile:
{profile_text}

Answers:
{answers}

Web signals:
{web_signals}

Identify the most important red flags and summarize them.
"""
    )
])
