from app.synthetic_data import make_candidate
from app.feature_engineering import build_feature_report


def test_feature_report_runs():
    c = make_candidate(1, fraud=True)
    fr = build_feature_report(c["profile_text"], c["answers"], c["web_signals"])
    assert 0.0 <= fr.timeline_anomaly <= 1.0
    assert 0.0 <= fr.answer_anomaly <= 1.0
    assert 0.0 <= fr.consistency_anomaly <= 1.0
