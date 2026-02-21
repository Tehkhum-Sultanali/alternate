import json
import re
import os
import random
from dotenv import load_dotenv
from openai import OpenAI

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
            "- Use bullets, probabilities, and tradeoffs.\n"
            "- Include a simple decision rubric.\n"
            "- Be blunt and practical.\n"
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
            "- Include 1 illegal in spirit move (NOT actually illegal), e.g. chaotic marketing stunt.\n"
        )
    return "VOICE RULES:\n- Be distinct and specific.\n"


SCHEMA_HINT = """
Return ONLY valid JSON with exactly these keys:
{
  "name": "string value here",
  "narrative": "string value here - use only single quotes or escaped double quotes inside",
  "headlines": ["string 1", "string 2", "string 3"],
  "strategy": "string value here - use only single quotes or escaped double quotes inside",
  "vulnerabilities": ["string 1", "string 2", "string 3"],
  "tone_score": 1.2,
  "risk_score": 0.8
}

CRITICAL JSON RULES:
- tone_score and risk_score MUST be plain numbers between 0.0 and 2.0 (no quotes)
- Do NOT use double quotes inside string values. Use single quotes instead.
- Do NOT use newlines inside string values. Write everything on one line.
- No trailing commas. No markdown. No code fences. No commentary.
- Output ONLY the JSON object, nothing else.
"""


def _safe_parse_json(content: str) -> dict:
    """Multi-strategy JSON parser that handles common LLM output issues."""
    # Strategy 1: Direct parse after stripping markdown fences
    cleaned = content.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Extract the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in output: {cleaned[:300]}")
    
    json_str = cleaned[start:end + 1]

    # Strategy 2: Try direct parse first (best case - model output is clean)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Fix trailing commas before } or ]
    repaired = re.sub(r',(\s*[}\]])', r'\1', json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Use json5-style liberal parsing via ast for simple cases,
    # or try replacing problematic characters
    # Replace unescaped newlines INSIDE string values only
    # We do this carefully: only replace \n that appear inside quoted strings
    def fix_newlines_in_strings(s):
        """Replace literal newlines inside JSON string values with spaces."""
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
                # Replace literal control chars inside strings with a space
                result.append(' ')
            else:
                result.append(ch)
        return ''.join(result)

    repaired2 = fix_newlines_in_strings(repaired)
    try:
        return json.loads(repaired2)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON after all repair attempts. Error: {e}. Content: {json_str[:400]}")


def _minimax_generate(agent_name: str, style: str, scenario: str, attempt: int = 1):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY (MiniMax key)")

    style_rules = _style_rules(agent_name)

    system = (
        f"You are {agent_name}. Follow the VOICE RULES exactly. "
        "You are generating a competitive 5-year counterfactual future packet. "
        "Be scenario-specific, not generic. "
        "IMPORTANT: Output MUST be a single valid JSON object ONLY. "
        "No markdown, no code fences, no commentary before or after the JSON. "
        "Do NOT use double quotes inside string values — use single quotes instead. "
        "Do NOT include literal newlines inside string values."
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
        temperature=0.9 if attempt == 1 else 0.5,  # Lower temp on retry
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("Empty model output (content is blank).")

    data = _safe_parse_json(content)

    # Validate required keys
    required = ["name", "narrative", "headlines", "strategy", "vulnerabilities", "tone_score", "risk_score"]
    for k in required:
        if k not in data:
            raise ValueError(f"LLM JSON missing key: {k}")

    # Coerce score types safely
    data["tone_score"] = float(data["tone_score"])
    data["risk_score"] = float(data["risk_score"])
    
    # Clamp scores to valid range
    data["tone_score"] = max(0.0, min(2.0, data["tone_score"]))
    data["risk_score"] = max(0.0, min(2.0, data["risk_score"]))

    data["name"] = agent_name
    data["source"] = "LLM"
    return data


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


def generate_futures(scenario: str):
    futures = []
    for a in AGENTS:
        if USE_LLM:
            result = None
            # Try up to 2 times before falling back to mock
            for attempt in range(1, 3):
                try:
                    result = _minimax_generate(a["name"], a["style"], scenario, attempt=attempt)
                    result["source"] = "LLM"
                    break
                except Exception as e:
                    if attempt == 2:
                        # Both attempts failed — use mock
                        result = _mock_future(a["name"], a["style"], scenario)
                        result["source"] = "MOCK"
                        result["error"] = f"{type(e).__name__}: {str(e)}"
            futures.append(result)
        else:
            mock = _mock_future(a["name"], a["style"], scenario)
            mock["source"] = "MOCK"
            futures.append(mock)
    return futures