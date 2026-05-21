"""
ContextGuard — Streamlit Dashboard

Visualises experiment results stored in results/sample_results.json.

Run with:
    streamlit run dashboard/streamlit_app.py
"""

import json
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextGuard Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 ContextGuard: LLM Reliability Dashboard")
st.caption("Evaluating hallucination sensitivity, context robustness, and response stability across prompt variants.")

# ── Load data ──────────────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "sample_results.json",
)

@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

try:
    df = load_results(RESULTS_PATH)
except FileNotFoundError:
    st.error(f"Results file not found at `{RESULTS_PATH}`. Run `python experiments/run_experiment.py` first.")
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

variants = sorted(df["variant"].unique().tolist())
selected_variants = st.sidebar.multiselect("Prompt Variants", variants, default=variants)

questions = sorted(df["question"].unique().tolist())
selected_questions = st.sidebar.multiselect("Questions", questions, default=questions)

filtered = df[
    df["variant"].isin(selected_variants) &
    df["question"].isin(selected_questions)
]

# ── Top-level KPIs ─────────────────────────────────────────────────────────────
st.subheader("📊 Aggregate Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Evaluations", len(filtered))
col2.metric("Avg Hallucination Score", round(filtered["hallucination_score"].mean(), 3))
col3.metric("Avg Length Penalty", round(filtered["length_penalty"].mean(), 3))
col4.metric("Avg Confidence Score", round(filtered["confidence_score"].mean(), 3))

st.divider()

# ── Variant comparison ─────────────────────────────────────────────────────────
st.subheader("📈 Variant Comparison")

variant_stats = (
    filtered.groupby("variant")[["hallucination_score", "length_penalty", "confidence_score"]]
    .mean()
    .round(4)
)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    fig, ax = plt.subplots(figsize=(8, 4))
    variant_stats.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Mean Scores by Prompt Variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig)

with col_table:
    st.dataframe(variant_stats, use_container_width=True)

st.divider()

# ── Per-question breakdown ─────────────────────────────────────────────────────
st.subheader("🔍 Per-Question Hallucination Scores")

q_stats = (
    filtered.groupby(["question", "variant"])["hallucination_score"]
    .mean()
    .unstack(fill_value=0)
    .round(3)
)
st.dataframe(q_stats, use_container_width=True)

st.divider()

# ── Raw results ────────────────────────────────────────────────────────────────
st.subheader("📋 Raw Results")

display_cols = ["question", "variant", "hallucination_score", "length_penalty", "confidence_score", "word_count"]
st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

with st.expander("Show full responses"):
    for _, row in filtered.iterrows():
        st.markdown(f"**[{row['variant']}]** *{row['question']}*")
        st.write(row["response"])
        st.divider()
