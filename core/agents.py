import json
import re
import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

USE_LLM = True

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.minimax.io/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

AGENTS = [
    {"name": "Visionary", "style": "optimistic, exponential, bold, big bets"},
    {"name": "Realist", "style": "cautious, grounded, risk-aware, pragmatic"},
    {"name": "Capitalist", "style": "profit-maximizing, distribution-first, leverage-driven"},
    {"name": "Chaos Agent", "style": "disruptive, attention-hacking, high volatility, unpredictable"},
]


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
            "- Be blunt and practical. Give a YES or NO verdict.\n"
            "- State 1 key risk and 1 key opportunity.\n"
            "- Keep it tight — no bullet lists, just dense prose.\n"
        )
    if agent_name == "Capitalist":
        return (
            "VOICE RULES:\n"
            "- Write like a VC / growth lead.\n"
            "- Talk in ROI, upside, downside, optionality.\n"
            "- Include a terms to demand section.\n"
            "- Use business metrics language (runway, dilution, moat).\n"
        )
    if agent_name == "Chaos Agent":
        return (
            "VOICE RULES:\n"
            "- Write like an unhinged hype strategist.\n"
            "- Use short punchy lines. Occasional caps.\n"
            "- Include 1 meme-y line.\n"
            "- Include 1 illegal in spirit move (NOT actually illegal).\n"
        )
    return "VOICE RULES:\n- Be distinct and specific.\n"


# Tighter schema with explicit char limits to prevent runaway output
SCHEMA_HINT = """
Return ONLY a valid JSON object with EXACTLY these keys. No other text.

{
  "name": "agent name",
  "narrative": "Max 300 chars. One paragraph. Clear YES or NO verdict first. No line breaks. No internal quotes.",
  "headlines": ["Under 80 chars each", "Under 80 chars each", "Under 80 chars each"],
  "strategy": "Max 300 chars. 3 actions separated by semicolons. No line breaks. No internal quotes.",
  "vulnerabilities": ["Under 80 chars each", "Under 80 chars each", "Under 80 chars each"],
  "tone_score": 1.2,
  "risk_score": 0.8
}

HARD RULES — violating any of these will cause a system failure:
1. tone_score and risk_score are plain floats between 0.0 and 2.0. Never strings.
2. narrative and strategy must be single-line strings under 300 characters each.
3. All array items must be single-line strings under 80 characters each.
4. No double quotes inside any string value. Use single quotes if needed.
5. No trailing commas. No markdown. No code fences. No commentary.
6. Output the JSON object and absolutely nothing else.
"""


def _safe_parse_json(content: str) -> dict:
    """Multi-strategy JSON parser. Sets _recovered=True if repairs were needed."""
    cleaned = content.strip()
    # Strip markdown fences
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Extract outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in output: {cleaned[:300]}")

    json_str = cleaned[start:end + 1]

    # Strategy 1: direct parse
    try:
        parsed = json.loads(json_str)
        parsed["_recovered"] = False
        return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: remove trailing commas
    repaired = re.sub(r',(\s*[}\]])', r'\1', json_str)
    try:
        parsed = json.loads(repaired)
        parsed["_recovered"] = True
        return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 3: replace literal control chars inside quoted strings only
    def fix_newlines_in_strings(s):
        result = []
        in_string = False
        escape_next = False
        for ch in s:
            if escape_next:
                result.append(ch)
                escape_next = False
            elif ch == '\\' and in_string:
                result.append(ch)
                escape_next = True
            elif ch == '"':
                in_string = not in_string
                result.append(ch)
            elif ch in ('\n', '\r', '\t') and in_string:
                result.append(' ')
            else:
                result.append(ch)
        return ''.join(result)

    repaired2 = fix_newlines_in_strings(repaired)
    try:
        parsed = json.loads(repaired2)
        parsed["_recovered"] = True
        return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 4: truncate to last valid closing brace
    # Sometimes the model outputs valid JSON followed by trailing garbage
    for end_idx in range(len(repaired2) - 1, 0, -1):
        if repaired2[end_idx] == '}':
            try:
                parsed = json.loads(repaired2[:end_idx + 1])
                parsed["_recovered"] = True
                return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"Could not parse JSON after all repair attempts. Content: {json_str[:400]}"
    )


def _minimax_generate(agent_name: str, style: str, scenario: str, attempt: int = 1):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY (MiniMax key)")

    style_rules = _style_rules(agent_name)

    system = (
        f"You are {agent_name}, a strategic persona generating a future scenario packet. "
        "OUTPUT FORMAT: You must return a single valid JSON object and absolutely nothing else. "
        "No markdown. No code fences. No explanation before or after. "
        "Keep all string values short and on a single line. "
        "Do NOT use double quotes inside string values."
    )

    user = (
        f"Scenario: {scenario}\n\n"
        f"Your persona: {agent_name} — {style}\n\n"
        f"{style_rules}\n"
        f"{SCHEMA_HINT}"
    )

    resp = client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.85 if attempt == 1 else 0.4,
        max_tokens=700,  # hard ceiling — prevents runaway long outputs that break JSON
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("Empty model output.")

    data = _safe_parse_json(content)

    required = ["name", "narrative", "headlines", "strategy", "vulnerabilities", "tone_score", "risk_score"]
    for k in required:
        if k not in data:
            raise ValueError(f"LLM JSON missing key: {k}")

    data["tone_score"] = max(0.0, min(2.0, float(data["tone_score"])))
    data["risk_score"] = max(0.0, min(2.0, float(data["risk_score"])))
    data["name"] = agent_name
    return data


def _mock_future(agent_name: str, style: str, scenario: str):
    tone = random.uniform(0.7, 1.5)
    risk = random.uniform(0.3, 1.7)
    narrative = (
        f"{agent_name} view: {scenario}. "
        f"Approach: {style}. This timeline makes decisions aligned with that ideology."
    )
    headlines = [
        f"[Year 1] {agent_name} reframes the scenario and sets the agenda",
        f"[Year 3] {agent_name} triggers a major inflection point",
        f"[Year 5] {agent_name} becomes the dominant narrative or collapses",
    ]
    strategy = (
        f"1) Immediate move aligned with {style}; "
        f"2) Mid-term compounding play; "
        f"3) Endgame: lock-in dominance or exit before collapse"
    )
    vulnerabilities = (
        ["High collapse probability", "Backlash / regulation risk", "Unstable coalition / trust decay"]
        if agent_name == "Chaos Agent"
        else ["Overconfidence risk", "Execution bottlenecks", "Second-order effects underestimated"]
    )
    return {
        "name": agent_name,
        "narrative": narrative,
        "headlines": headlines,
        "strategy": strategy,
        "vulnerabilities": vulnerabilities,
        "tone_score": tone,
        "risk_score": risk,
    }


def _one_agent_future(a: dict, scenario: str) -> dict:
    if not USE_LLM:
        result = _mock_future(a["name"], a["style"], scenario)
        result["source"] = "Fallback"
        result["meta"] = {"attempts_used": 0}
        return result

    last_err = None
    scenario_low = (scenario or "").strip().lower()

    for attempt in range(1, 3):
        try:
            result = _minimax_generate(a["name"], a["style"], scenario, attempt=attempt)

            # Echo guard — if the model just repeated the scenario, reject it
            narrative_low = (result.get("narrative") or "").strip().lower()
            if scenario_low and narrative_low and scenario_low[:80] in narrative_low:
                raise ValueError("Model echoed the scenario instead of analysing it.")

            recovered = bool(result.pop("_recovered", False))
            result["source"] = f"MiniMax (recovered)" if recovered else "MiniMax"
            result["meta"] = {"attempts_used": attempt}
            return result

        except Exception as e:
            last_err = e

    # Both attempts failed — use mock
    result = _mock_future(a["name"], a["style"], scenario)
    result["source"] = "Fallback"
    result["error"] = f"{type(last_err).__name__}: {str(last_err)}" if last_err else "Unknown"
    result["meta"] = {"attempts_used": 2}
    return result


def generate_futures(scenario: str) -> list:
    futures = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        jobs = {ex.submit(_one_agent_future, a, scenario): a for a in AGENTS}
        for job in as_completed(jobs):
            futures.append(job.result())

    # Preserve consistent display order: Visionary, Realist, Capitalist, Chaos Agent
    order = {a["name"]: i for i, a in enumerate(AGENTS)}
    futures.sort(key=lambda x: order.get(x.get("name", ""), 999))
    return futures



def generate_one_future(agent_name: str, scenario: str):
    """
    Generate exactly one agent future (with retry + fallback), so Streamlit
    can stream progress per-agent.
    """
    agent = next((a for a in AGENTS if a["name"] == agent_name), None)
    if not agent:
        raise ValueError(f"Unknown agent: {agent_name}")

    if USE_LLM:
        for attempt in range(1, 3):
            try:
                return _minimax_generate(agent["name"], agent["style"], scenario, attempt=attempt)
            except Exception as e:
                if attempt == 2:
                    mock = _mock_future(agent["name"], agent["style"], scenario)
                    mock["error"] = f"{type(e).__name__}: {str(e)}"
                    return mock
    else:
        return _mock_future(agent["name"], agent["style"], scenario)