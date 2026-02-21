def score_futures(futures):
    scores = []
    for f in futures:
        tone = f["tone_score"]
        risk = f["risk_score"]

        # Influence likes boldness (tone) but hates instability (risk)
        influence = (1.2 * tone) - (0.35 * risk)

        # Stability hates risk
        stability = max(0.05, 1.35 - (0.6 * risk))

        # Risk is risk
        risk_out = risk

        # Small bonus: more coherent plans get a nudge
        plan_len = len(f.get("strategy", ""))
        influence += min(0.15, plan_len / 8000)

        scores.append({
            "name": f["name"],
            "influence": max(0.0, influence),
            "stability": max(0.0, stability),
            "risk": max(0.0, risk_out),
        })
    return scores 