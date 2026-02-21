import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from core.agents import generate_futures
from core.scoring import score_futures
from core.simulation import simulate_trajectories

st.set_page_config(page_title="AI War Room", layout="wide")

st.title("⚔️ AI War Room")
st.caption("Generative Strategy Battle Engine (MVP skeleton — sponsor APIs plug in next).")

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

if st.button("Run War Room", type="primary"):
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
            st.write(f["summary"])
            st.metric("Influence", f"{s['influence']:.2f}")
            st.metric("Stability", f"{s['stability']:.2f}")
            st.metric("Risk", f"{s['risk']:.2f}")

    fig, ax = plt.subplots()
    for name, curve in trajectories.items():
        ax.plot(curve, label=name)
    ax.set_title("Influence trajectory over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Influence")
    ax.legend()
    st.pyplot(fig)

    final = [(name, curve[-1]) for name, curve in trajectories.items()]
    winner_name, winner_val = max(final, key=lambda x: x[1])

    st.success(f"🏆 Winning future: **{winner_name}** (final influence = {winner_val:.2f})")