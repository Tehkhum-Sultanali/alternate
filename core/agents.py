import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Flip True when you have a key; keep True and it will fallback safely if missing.
USE_LLM = True

# MiniMax (OpenAI-compatible)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.minimax.io/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

AGENTS = [
    {"name": "Visionary", "style": "optimistic, exponential, bold, big bets"},
    {"name": "Realist", "style": "cautious, grounded, risk-aware, pragmatic"},
    {"name": "Capitalist", "style": "profit-maximizing, distribution-first, leverage-driven"},
    {"name": "Chaos Agent", "style": "disruptive, attention-hacking, high volatility, unpredictable"},
]
# adding Style rules 

def _style_rules(agent_name: str):
    if agent_name == "Visionary":
        return (
            "VOICE RULES:\n"
            "- Write like a founder-pitch + futurist.\n"
            "- Use vivid metaphors + decisive language.\n"
            "- Include 1 bold contrarian claim.\n"
            "- Avoid generic advice. Be specific.\n"
        )
    if agent_name == "Realist":
        return (
            "VOICE RULES:\n"
            "- Write like an operator + risk manager.\n"
            "- Use bullets, probabilities, and tradeoffs.\n"
            "- Include a simple decision rubric.\n"
            "- Be blunt and practical.\n"
        )
    if agent_name == "Capitalist":
        return (
            "VOICE RULES:\n"
            "- Write like a VC / growth lead.\n"
            "- Talk in ROI, upside, downside, optionality.\n"
            "- Include a ‘terms to demand’ section.\n"
            "- Use business metrics language (runway, dilution, moat).\n"
        )
    if agent_name == "Chaos Agent":
        return (
            "VOICE RULES:\n"
            "- Write like an unhinged hype strategist.\n"
            "- Use short punchy lines. Occasional caps.\n"
            "- Include 1 meme-y line.\n"
            "- Include 1 ‘illegal in spirit’ move (NOT actually illegal), e.g. chaotic marketing stunt.\n"
        )
    return "VOICE RULES:\n- Be distinct and specific.\n"



SCHEMA_HINT = """
Return ONLY valid JSON with exactly these keys:
{
  "name": string,
  "narrative": string,
  "headlines": [string, string, string],
  "strategy": string,
  "vulnerabilities": [string, string, string],
  "tone_score": number,
  "risk_score": number
}
No extra keys. No markdown. No commentary.
tone_score and risk_score must be in [0.0, 2.0].
"""

def _mock_future(agent_name: str, style: str, scenario: str):
    tone = random.uniform(0.7, 1.5)
    risk = random.uniform(0.3, 1.7)

    narrative = (
        f"{agent_name} view: {scenario}\n\n"
        f"Approach: {style}. This timeline makes decisions aligned with that ideology."
    )
    headlines = [
        f"[Year 1] {agent_name} reframes the scenario and sets the agenda",
        f"[Year 3] {agent_name} triggers a major inflection point",
        f"[Year 5] {agent_name} becomes the dominant narrative (or collapses)",
    ]
    strategy = (
        f"3-step strategy:\n"
        f"1) Immediate move aligned with {style}\n"
        f"2) Mid-term compounding play (distribution + capability)\n"
        f"3) Endgame: lock-in dominance or exit before collapse"
    )
    vulnerabilities = [
        "Overconfidence risk",
        "Execution bottlenecks",
        "Second-order effects underestimated",
    ]
    if agent_name == "Chaos Agent":
        vulnerabilities = [
            "High collapse probability",
            "Backlash / regulation risk",
            "Unstable coalition / trust decay",
        ]

    return {
        "name": agent_name,
        "narrative": narrative,
        "headlines": headlines,
        "strategy": strategy,
        "vulnerabilities": vulnerabilities,
        "tone_score": tone,
        "risk_score": risk,
    }

def _minimax_generate(agent_name: str, style: str, scenario: str):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY (MiniMax key)")

    style_rules = _style_rules(agent_name)

    system = (
        f"You are {agent_name}. Follow the VOICE RULES exactly. "
        "You are generating a competitive 5-year counterfactual future packet. "
        "Be scenario-specific, not generic. Output must be strict JSON only."
    )

    user = f"""
Scenario: {scenario}

Agent persona:
- name: {agent_name}
- style: {style}

{style_rules}

HARD REQUIREMENTS:
- Narrative must contain a clear YES/NO verdict and why.
- Strategy must list 3 concrete actions starting tomorrow morning.
- Vulnerabilities must be specific to this scenario (no generic fluff).
- Headlines must be punchy and distinct.

{SCHEMA_HINT}
"""

    resp = client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=1.0,
    )

    content = resp.choices[0].message.content.strip()

    # Robust JSON extraction (handles extra text around JSON)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output. Raw: {content[:200]}")

    data = json.loads(content[start:end + 1])

    required = ["name", "narrative", "headlines", "strategy", "vulnerabilities", "tone_score", "risk_score"]
    for k in required:
        if k not in data:
            raise ValueError(f"LLM JSON missing key: {k}")

    data["name"] = agent_name  # lock name to persona
    data["source"] = "LLM"
    return data

def generate_futures(scenario: str):
    futures = []
    for a in AGENTS:
        if USE_LLM:
            try:
                data = _minimax_generate(a["name"], a["style"], scenario)
                # (LLM already sets data["source"] = "LLM" in our updated _minimax_generate,
                # but leaving this line is harmless if you want redundancy)
                data["source"] = data.get("source", "LLM")
                futures.append(data)
            except Exception as e:
                mock = _mock_future(a["name"], a["style"], scenario)
                mock["source"] = "MOCK"
                mock["error"] = f"{type(e).__name__}: {str(e)}"
                futures.append(mock)
        else:
            mock = _mock_future(a["name"], a["style"], scenario)
            mock["source"] = "MOCK"
            futures.append(mock)
    return futures