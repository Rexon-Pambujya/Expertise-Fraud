from app.synthetic_data import build_dataset
from app.env import ExpertiseFraudEnv


def test_env_step():
    data = build_dataset(10)
    env = ExpertiseFraudEnv(data)
    obs, info = env.reset()
    assert len(obs) == 4
    obs2, reward, terminated, truncated, info2 = env.step(2)
    assert len(obs2) == 4
