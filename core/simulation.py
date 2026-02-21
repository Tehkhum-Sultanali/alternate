import numpy as np

def simulate_trajectories(scores, steps: int = 24):
    """
    Produce simple trajectories for Influence over time.
    Higher risk -> higher volatility.
    Higher stability -> smoother curve.
    """
    trajectories = {}
    for s in scores:
        base = s["influence"]
        vol = 0.12 + 0.18 * min(1.0, s["risk"] / 2.0)
        drift = 0.03 + 0.05 * min(1.0, s["stability"])

        values = [base]
        for _ in range(steps - 1):
            step = values[-1] + drift + np.random.normal(0, vol)
            values.append(max(0.0, step))
        trajectories[s["name"]] = np.array(values)
    return trajectories