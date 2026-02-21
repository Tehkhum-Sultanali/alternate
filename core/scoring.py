import random

def score_futures(futures):
    scores = []
    for f in futures:
        influence = random.uniform(0.6, 1.6)
        stability = random.uniform(0.3, 1.0)
        risk = random.uniform(0.2, 1.7)

        # Make Chaos riskier but sometimes more influential
        if "Chaos" in f["name"]:
            influence += 0.2
            risk += 0.3
            stability -= 0.1

        scores.append({
            "name": f["name"],
            "influence": max(0.0, influence),
            "stability": max(0.0, stability),
            "risk": max(0.0, risk),
        })
    return scores