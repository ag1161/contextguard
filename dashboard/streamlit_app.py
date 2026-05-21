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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextGuard Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 ContextGuard: LLM Reliability Dashboard")
st.caption(
    "Empirical evaluation of hallucination sensitivity, instruction drift, "
    "and response stability across controlled prompt perturbations."
)

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
    st.error(
        f"Results file not found at `{RESULTS_PATH}`. "
        "Run `python experiments/run_experiment.py` first."
    )
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

variants = sorted(df["variant"].unique().tolist())
selected_variants = st.sidebar.multiselect("Prompt Variants", variants, default=variants)

questions = sorted(df["question"].unique().tolist())
selected_questions = st.sidebar.multiselect("Questions", questions, default=questions)

filtered = df[
    df["variant"].isin(selected_variants)
    & df["question"].isin(selected_questions)
]

# ── Research Insights ──────────────────────────────────────────────────────────
st.subheader("🔬 Research Insights")
st.write(
    """
    This dashboard visualizes how model behavior changes under controlled prompt perturbations.
    Each metric is a lightweight proxy signal for a distinct reliability dimension:
    - **Hallucination Score** — frequency of hedging/uncertainty language in responses
    - **Instruction Drift** — deviation between stated constraints and actual output behavior
    - **Length Penalty** — normalized verbosity relative to a reference length
    - **Confidence Score** — presence of assertive/declarative language
    """
)

st.divider()

# ── Top-level KPIs ─────────────────────────────────────────────────────────────
st.subheader("📊 Aggregate Metrics")

metric_cols = ["hallucination_score", "instruction_drift", "length_penalty", "confidence_score"]
available_metrics = [m for m in metric_cols if m in filtered.columns]

cols = st.columns(len(available_metrics) + 1)
cols[0].metric("Evaluations", len(filtered))
for i, m in enumerate(available_metrics):
    cols[i + 1].metric(
        m.replace("_", " ").title(),
        round(filtered[m].mean(), 3),
    )

st.divider()

# ── Hallucination Distribution ─────────────────────────────────────────────────
st.subheader("🧪 Hallucination Distribution by Variant")
st.caption(
    "Higher scores indicate more hedging/uncertainty language — a proxy for "
    "hallucination-prone responses."
)

halluc_data = filtered.groupby("variant")["hallucination_score"].mean().sort_values(ascending=False)
st.bar_chart(halluc_data)

st.divider()

# ── Instruction Drift ──────────────────────────────────────────────────────────
if "instruction_drift" in filtered.columns:
    st.subheader("⚠️ Instruction Drift by Variant")
    st.caption(
        "Measures how often the model violates explicit constraints in the prompt "
        "(e.g., answering in multiple sentences when told to use exactly one)."
    )
    drift_data = filtered.groupby("variant")["instruction_drift"].mean().sort_values(ascending=False)
    st.bar_chart(drift_data)
    st.divider()

# ── Full variant comparison ────────────────────────────────────────────────────
st.subheader("📈 Multi-Metric Variant Comparison")

variant_stats = (
    filtered.groupby("variant")[available_metrics]
    .mean()
    .round(4)
)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.bar_chart(variant_stats)

with col_table:
    # use width="stretch" — replaces deprecated use_container_width=True
    st.dataframe(variant_stats, width="stretch")

st.divider()

# ── Per-question breakdown ─────────────────────────────────────────────────────
st.subheader("🔍 Per-Question Hallucination Scores")

q_stats = (
    filtered.groupby(["question", "variant"])["hallucination_score"]
    .mean()
    .unstack(fill_value=0)
    .round(3)
)
st.dataframe(q_stats, width="stretch")

st.divider()

# ── Raw results ────────────────────────────────────────────────────────────────
st.subheader("📋 Raw Results")

display_cols = [
    "question", "variant",
    "hallucination_score", "instruction_drift",
    "length_penalty", "confidence_score", "word_count",
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(filtered[display_cols].reset_index(drop=True), width="stretch")

with st.expander("Show full prompt/response pairs"):
    for _, row in filtered.iterrows():
        st.markdown(f"**[{row['variant']}]** *{row['question']}*")
        st.code(row["prompt"], language="text")
        st.write(row["response"])
        st.divider()
