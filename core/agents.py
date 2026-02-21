def generate_futures(scenario: str):
    # Mock "generative" outputs for now — we will swap this to sponsor LLM calls later
    return [
        {
            "name": "Visionary",
            "summary": f"Visionary future: embraces exponential growth and bold bets in response to: {scenario}",
        },
        {
            "name": "Realist",
            "summary": f"Realist future: prioritizes stability, risk control, and incremental wins around: {scenario}",
        },
        {
            "name": "Capitalist",
            "summary": f"Capitalist future: optimizes monetization, distribution, and leverage under: {scenario}",
        },
        {
            "name": "Chaos Agent",
            "summary": f"Chaos future: maximizes disruption and attention, accepting instability around: {scenario}",
        },
    ]