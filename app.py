import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from core.agents import generate_futures
from core.scoring import score_futures
from core.simulation import simulate_trajectories

st.set_page_config(page_title="Alternate", layout="wide")

st.title("Alternate")
st.caption("A generative strategy battle engine. (MVP now — sponsor APIs plug in next)")

scenario = st.text_area(
    "Enter a high-stakes scenario:",
    placeholder="e.g., A new open-source video model drops tomorrow and disrupts the market…",
    height=120
)

colA, colB = st.columns([1, 1])
with colA:
    steps = st.slider("Simulation horizon (months)", 12, 60, 24, 6)
with colB:
    seed = st.number_input("Random seed (demo stability)", value=7, step=1)

if st.button("Run Alternate", type="primary"):
    if not scenario.strip():
        st.warning("Type a scenario first.")
        st.stop()

    np.random.seed(int(seed))

    futures = generate_futures(scenario)
    scores = score_futures(futures)
    trajectories = simulate_trajectories(scores, steps=steps)

    st.divider()

    cols = st.columns(4)
    for i, f in enumerate(futures):
        s = scores[i]
        with cols[i]:
            st.subheader(f["name"])
            st.caption(f"Source: {f.get('source','UNKNOWN')}")
            if f.get("error"):
                st.error(f.get("error"))
            # quick one-line preview (so the cards don't get massive)
            preview = f.get("narrative", "").split("\n")[0]
            st.write(preview)

            st.metric("Influence", f"{s['influence']:.2f}")
            st.metric("Stability", f"{s['stability']:.2f}")
            st.metric("Risk", f"{s['risk']:.2f}")

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

    # Plot influence curves
    fig, ax = plt.subplots()
    for name, curve in trajectories.items():
        ax.plot(curve, label=name)
    ax.set_title("Influence trajectory over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Influence")
    ax.legend()
    st.pyplot(fig)

    # Winner selection (highest final influence)
    final = [(name, curve[-1]) for name, curve in trajectories.items()]
    winner_name, winner_val = max(final, key=lambda x: x[1])

    st.success(f"Winning future: **{winner_name}** (final influence = {winner_val:.2f})")

    # Battle verdict (MVP theatrics; later LLM can generate these attacks)
    st.subheader("Battle Verdict (MVP)")

    roasts = {
        "Visionary": "You assume the world cooperates. It won’t.",
        "Realist": "You stabilize so hard you miss the breakout moment.",
        "Capitalist": "You monetize fast and lose legitimacy. Enjoy the backlash.",
        "Chaos Agent": "You go viral and then implode. Classic.",
    }

    st.write("**Cross-attacks:**")
    for s in scores:
        st.write(f"- **{s['name']}** to everyone else: {roasts.get(s['name'], 'I outplay you.')}")