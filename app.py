import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from core.agents import generate_futures
from core.scoring import score_futures
from core.simulation import simulate_trajectories

import os
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")

st.set_page_config(page_title="Alternate", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Hero banner */
.alt-hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #111827 60%, #1a0a2e 100%);
    border: 1px solid #2a2a3e;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.alt-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.alt-hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -1px;
}
.alt-hero .tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #8b5cf6;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.alt-hero .description {
    color: #9ca3af;
    font-size: 1rem;
    line-height: 1.7;
    max-width: none;
}
.alt-hero .description strong {
    color: #e5e7eb;
}

/* Pill badges */
.pill-row {
    display: flex;
    gap: 10px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.pill {
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.35);
    color: #a78bfa;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 12px;
    border-radius: 99px;
    letter-spacing: 1px;
}

/* Winner banner */
.winner-banner {
    background: linear-gradient(135deg, #0d1f0d, #1a2e1a);
    border: 1.5px solid #22c55e;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin: 1.5rem 0;
    position: relative;
}
.winner-banner .crown {
    font-size: 2rem;
    margin-bottom: 0.3rem;
}
.winner-banner .winner-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #22c55e;
    margin: 0;
}
.winner-banner .winner-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #4ade80;
    margin-bottom: 0.8rem;
}
.winner-banner .winner-reason {
    color: #d1fae5;
    font-size: 0.95rem;
    line-height: 1.6;
    border-top: 1px solid rgba(34,197,94,0.2);
    padding-top: 0.8rem;
    margin-top: 0.5rem;
}

/* Metric tooltip rows */
.metric-row {
    margin-bottom: 0.5rem;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1px;
}
.metric-desc {
    font-size: 0.72rem;
    color: #4b5563;
    margin-bottom: 4px;
    font-style: italic;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f3f4f6;
}
.metric-bar {
    height: 4px;
    border-radius: 2px;
    background: #1f2937;
    margin-top: 4px;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* Agent card header */
.agent-header {
    margin-bottom: 0.6rem;
}
.agent-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    color: #f9fafb;
}
.agent-source {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b7280;
    margin-top: 2px;
}

/* Verdict section */
.verdict-box {
    background: #0d0d14;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
}
.verdict-box .agent-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8b5cf6;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.verdict-box .roast {
    color: #d1d5db;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ── Agent Explainer Cards (matches hero aesthetic) ───────────────────────── */
.model-row{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 0.5rem 0 2rem 0;
}

.model-card{
    background: rgba(17, 24, 39, 0.55);
    border: 1px solid rgba(42, 42, 62, 0.95);
    border-radius: 16px;
    padding: 16px 16px 14px 16px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    transition: transform .18s ease, border-color .18s ease, background .18s ease;
}

.model-card::before{
    content: "";
    position: absolute;
    top: -70px;
    right: -70px;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,92,246,0.16) 0%, transparent 70%);
    pointer-events: none;
}

.model-card:hover{
    transform: translateY(-2px);
    border-color: rgba(139,92,246,0.55);
    background: rgba(17, 24, 39, 0.75);
}

.model-title{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: #f9fafb;
    margin-bottom: 8px;
    letter-spacing: -0.2px;
}

.model-title .icon{
    font-family: 'Space Mono', monospace;
    color: #a78bfa;
    margin-right: 8px;
}

.model-desc{
    color: #9ca3af;
    font-size: 0.9rem;
    line-height: 1.45;
}

@media (max-width: 1100px){
    .model-row{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 650px){
    .model-row{ grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="alt-hero">
    <div class="tagline">⚡ Generative Strategy Engine</div>
    <h1>Alternate</h1>
    <div class="description">
        Describe a high-stakes scenario. Four ideological agents <strong>Visionary, Realist, Capitalist, Chaos Agent.</strong> Each generating a competing future.<br><br>
        No advice. No consensus. <strong>Parallel realities, battling for dominance.</strong><br>
        Watch their influence curves diverge over time. Only one worldview wins.
    </div>
    <div class="pill-row">
        <span class="pill">4 Competing Agents</span>
        <span class="pill">Counterfactual Futures</span>
        <span class="pill">Influence Simulation</span>
        <span class="pill">Strategy Packets</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Agent Explainer Cards (RIGHT AFTER HERO) ──────────────────────────────────
st.markdown("""
<div class="model-row">
  <div class="model-card">
    <div class="model-title"></span>Visionary</div>
    <div class="model-desc">Thinks in long-term asymmetry and exponential upside, prioritizing bold moves that reshape the narrative.</div>
  </div>

  <div class="model-card">
    <div class="model-title"></span>Realist</div>
    <div class="model-desc">Optimizes for downside protection, operational feasibility, and survivability under uncertainty.</div>
  </div>

  <div class="model-card">
    <div class="model-title"></span>Capitalist</div>
    <div class="model-desc">Maximizes leverage, optionality, and economic value while negotiating for structural advantage.</div>
  </div>

  <div class="model-card">
    <div class="model-title"></span>Chaos Agent</div>
    <div class="model-desc">Exploits attention, volatility, and asymmetric disruption to force momentum through controlled instability.</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
scenario = st.text_area(
    "Enter a high-stakes scenario:",
    placeholder="e.g., A new open-source video model drops tomorrow and disrupts the market…",
    height=120
)

if st.button("Run Alternate", type="primary"):
    if not scenario.strip():
        st.warning("Type a scenario first.")
        st.stop()

    np.random.seed(42)  # fixed seed — no user control needed

    with st.spinner("Generating parallel futures..."):
        futures = generate_futures(scenario)
    scores = score_futures(futures)
    trajectories = simulate_trajectories(scores, steps=24)  # fixed 24-month horizon

    st.divider()

    # ── Winner (computed BEFORE agent cards so it shows on top) ──────────────
    final = [(name, curve[-1]) for name, curve in trajectories.items()]
    winner_name, winner_val = max(final, key=lambda x: x[1])

    # Find winner's data for the reason
    winner_future = next((f for f in futures if f["name"] == winner_name), {})
    winner_score_data = next((s for s in scores if s["name"] == winner_name), {})

    # Build a dynamic reason from the scores
    inf = winner_score_data.get("influence", 0)
    stab = winner_score_data.get("stability", 0)
    risk = winner_score_data.get("risk", 0)

    if inf > 1.5:
        inf_reason = "commanding influence"
    elif inf > 1.0:
        inf_reason = "strong influence"
    else:
        inf_reason = "steady influence"

    if stab > 0.9:
        stab_reason = "high structural stability"
    elif stab > 0.6:
        stab_reason = "moderate stability"
    else:
        stab_reason = "volatile but potent energy"

    if risk < 0.6:
        risk_reason = "low exposure to downside"
    elif risk < 1.2:
        risk_reason = "calculated risk"
    else:
        risk_reason = "high risk tolerance that paid off here"

    winner_headline = winner_future.get("headlines", [""])[0] if winner_future.get("headlines") else ""

    reason_text = (
        f"{winner_name} emerged dominant with {inf_reason} (↑{inf:.2f}), "
        f"{stab_reason} (⬡{stab:.2f}), and {risk_reason} (⚠ {risk:.2f}). "
        f"Over the 24-month simulation horizon, its trajectory compounded furthest. "
    )
    if winner_headline:
        reason_text += f'Key signal: <em>"{winner_headline}"</em>'

    st.markdown(f"""
    <div class="winner-banner">
        <div class="crown">🏆</div>
        <div class="winner-name">{winner_name} wins</div>
        <div class="winner-score">Final influence score: {winner_val:.2f}</div>
        <div class="winner-reason">{reason_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Influence Chart (below winner, above agent cards) ────────────────────
    AGENT_COLORS = {
        "Visionary": "#8b5cf6",
        "Realist": "#3b82f6",
        "Capitalist": "#f59e0b",
        "Chaos Agent": "#ef4444",
    }

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    for name, curve in trajectories.items():
        color = AGENT_COLORS.get(name, "#ffffff")
        ax.plot(curve, label=name, color=color, linewidth=2.2,
                alpha=1.0 if name == winner_name else 0.6)
        if name == winner_name:
            ax.plot(len(curve) - 1, curve[-1], "o", color=color, markersize=8)

    ax.set_title("Influence trajectory (24-month horizon)", color="#9ca3af",
                 fontsize=10, pad=10, loc="left")
    ax.set_xlabel("Month", color="#6b7280", fontsize=9)
    ax.set_ylabel("Influence", color="#6b7280", fontsize=9)
    ax.tick_params(colors="#4b5563")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2937")

    legend = ax.legend(framealpha=0, labelcolor="white", fontsize=9)
    st.pyplot(fig)

    st.markdown("""
    <p style="font-size:0.78rem; color:#4b5563; margin-top:-8px; margin-bottom:1.5rem;">
    📊 <em>The chart shows how each agent's influence compounds over time based on their tone and risk scores.
    A steeper curve = more aggressive compounding. Volatile agents can surge early but destabilize.
    The winning agent has the highest final influence at month 24.</em>
    </p>
    """, unsafe_allow_html=True)

    # ── Agent Cards ───────────────────────────────────────────────────────────
    cols = st.columns(4)

    METRIC_META = {
        "influence": {
            "label": "Influence",
            "desc": "How much this worldview bends the outcome over time. Higher = more dominant.",
            "color": "#8b5cf6",
            "max": 2.0,
        },
        "stability": {
            "label": "Stability",
            "desc": "How structurally sound the strategy is under pressure. Higher = less likely to collapse.",
            "color": "#22c55e",
            "max": 1.35,
        },
        "risk": {
            "label": "Risk",
            "desc": "Exposure to catastrophic failure or backfire. Higher = more volatile.",
            "color": "#ef4444",
            "max": 2.0,
        },
    }

    for i, f in enumerate(futures):
        s = scores[i]
        agent_name = f["name"]
        color = AGENT_COLORS.get(agent_name, "#ffffff")
        source_label = f.get("source", "Unknown")

        # Use actual model name from env if LLM-powered
        if "MiniMax" in source_label or source_label in ("LLM",):
            display_source = f"Model: {MINIMAX_MODEL}"
        elif source_label == "Fallback":
            display_source = "Source: Fallback (mock)"
        else:
            display_source = f"Source: {source_label}"

        with cols[i]:
            st.markdown(f"""
            <div class="agent-header">
                <div class="agent-name" style="border-left: 3px solid {color}; padding-left: 8px;">{agent_name}</div>
                <div class="agent-source">{display_source}</div>
            </div>
            """, unsafe_allow_html=True)

            if f.get("error"):
                st.error(f.get("error"))

            narrative = f.get("narrative", "")
            st.markdown(
                f'<p style="color:#9ca3af; font-size:0.88rem; line-height:1.6; margin-bottom:1rem;">{narrative}</p>',
                unsafe_allow_html=True
            )

            # Metrics with labels and mini progress bars
            for key in ["influence", "stability", "risk"]:
                meta = METRIC_META[key]
                val = s[key]
                pct = min(100, int((val / meta["max"]) * 100))
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-label">{meta['label']}</div>
                    <div class="metric-desc">{meta['desc']}</div>
                    <div class="metric-value">{val:.2f}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="width:{pct}%; background:{meta['color']};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("Open future packet", expanded=False):
                st.write("**Narrative**")
                st.write(f.get("narrative", ""))

                st.write("**Headlines**")
                for h in f.get("headlines", []):
                    st.write("-", h)

                st.write("**Strategy**")
                st.code(f.get("strategy", ""))

                st.write("**Vulnerabilities**")
                for v in f.get("vulnerabilities", []):
                    st.write("-", v)

    # ── Battle Verdict ────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="font-family: Space Mono, monospace; font-size:0.75rem; color:#8b5cf6; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;">⚔ Battle Verdict</p>',
        unsafe_allow_html=True
    )

    ranked = sorted(scores, key=lambda x: x["influence"], reverse=True)
    winner_s = ranked[0]
    loser_s = ranked[-1]

    dynamic_roasts = {
        "Visionary": f"You paint the future in bold strokes — but {loser_s['name']} called your bluff.",
        "Realist": f"You stabilised the situation well. But stability without momentum handed {winner_s['name']} the crown.",
        "Capitalist": f"You captured the upside early. {winner_s['name']} just had a bigger moat.",
        "Chaos Agent": f"You burned bright. Influence spiked. Then the structure ate you. {winner_s['name']} outlasted you.",
    }

    for s in scores:
        st.markdown(f"""
        <div class="verdict-box">
            <div class="agent-tag">{s['name']}</div>
            <div class="roast">{dynamic_roasts.get(s['name'], 'I outplay you all.')}</div>
        </div>
        """, unsafe_allow_html=True)