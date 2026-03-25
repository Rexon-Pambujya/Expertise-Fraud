from __future__ import annotations

from dataclasses import dataclass
import numpy as np


ACTIONS = {0: "PASS", 1: "FLAG", 2: "ASK_MORE"}


@dataclass
class HeuristicPolicy:
    flag_threshold: float = 0.7
    ask_threshold: float = 0.45

    def act(self, obs: np.ndarray) -> int:
        risk = float(np.mean(obs))
        if risk >= self.flag_threshold:
            return 1
        if risk >= self.ask_threshold:
            return 2
        return 0


def choose_action_from_report(report_dict: dict) -> str:
    risk = (
        0.35 * report_dict["timeline_anomaly"]
        + 0.30 * report_dict["answer_anomaly"]
        + 0.20 * report_dict["consistency_anomaly"]
        + 0.15 * report_dict["web_anomaly"]
    )
    if risk > 0.5:
        return "FLAG"
    if risk > 0.45:
        return "ASK_MORE"
    return "PASS"
