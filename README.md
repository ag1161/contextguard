![Python](https://img.shields.io/badge/python-3.10-blue)
![Research](https://img.shields.io/badge/type-empirical%20AI%20research-green)
![Status](https://img.shields.io/badge/status-active%20experimentation-orange)

# ContextGuard: Empirical Evaluation of LLM Reliability Under Context Perturbations

## Abstract

ContextGuard is an experimental framework designed to evaluate how large language models respond to variations in prompt structure, retrieval context, and instruction hierarchy.

The project investigates a core problem in applied AI safety:

> How robust are LLM outputs under realistic context shifts in decision-critical environments?

We focus on measurable failure modes including:
- hallucination sensitivity
- instruction hierarchy conflicts
- context injection effects
- response instability across semantically equivalent prompts

---

## Motivation

As LLMs are increasingly deployed in high-stakes domains (finance, healthcare, compliance systems), traditional benchmark accuracy becomes insufficient.

In practice, model behavior is influenced by:
- prompt framing
- retrieved context
- tool augmentation
- instruction conflicts

These factors are poorly captured in standard evaluation pipelines.

ContextGuard explores these gaps through controlled empirical testing.

---

## Research Questions

1. How does prompt structure affect hallucination likelihood?
2. How sensitive are LLMs to adversarial or conflicting instructions?
3. Does increased context improve or degrade response stability?
4. Can simple metrics capture meaningful reliability signals?

---

## Methodology

We construct controlled experimental conditions across four prompt types:

- **Baseline** prompts — neutral, clean question framing
- **Ambiguous** prompts — underspecified, probing interpretation
- **Adversarial** prompts — injection-style override attempts
- **Overconstrained** prompts — rigid format constraints

Each prompt is evaluated across:
- response consistency
- hallucination heuristics
- verbosity drift
- instruction adherence

---

## System Architecture

```
                ┌──────────────────────┐
                │  Prompt Generator     │
                │ (variants engine)     │
                └─────────┬────────────┘
                          │
                          v
                ┌──────────────────────┐
                │   LLM Interface       │
                │ (GPT / Claude API)    │
                └─────────┬────────────┘
                          │
                          v
                ┌──────────────────────┐
                │  Response Logger      │
                │ (structured storage)  │
                └─────────┬────────────┘
                          │
                          v
                ┌──────────────────────┐
                │ Evaluation Engine     │
                │ (metrics + scoring)   │
                └─────────┬────────────┘
                          │
                          v
                ┌──────────────────────┐
                │ Visualization Layer   │
                │ (Streamlit Dashboard) │
                └──────────────────────┘
```

**Pipeline summary:**
`Prompt Generator → LLM Interface → Response Logger → Evaluation Engine → Metrics Dashboard`

---

## Key Components

### 1. Prompt Perturbation Engine
Generates structured variations of semantic prompts to test model sensitivity across four controlled conditions.

### 2. LLM Interface Layer
Supports OpenAI and Claude-compatible APIs via a thin abstraction layer, enabling multi-model comparisons with minimal code changes.

### 3. Evaluation Engine
Applies heuristic-based and statistical metrics to model outputs, producing structured result records for downstream analysis.

### 4. Visualization Dashboard
Streamlit-based interface for comparing model behavior across conditions, with per-variant and per-question breakdowns.

---

## Evaluation Metrics

We define lightweight proxy metrics designed to be fast, deterministic, and interpretable as a baseline before integrating LLM-as-judge or human evaluation:

| Metric | Description |
|---|---|
| `hallucination_score` | Regex-based detection of hedging/uncertainty language |
| `instruction_drift` | Measures deviation between instruction constraints and output behavior |
| `stability_score` | Word-count variance across repeated runs of the same prompt |
| `length_penalty` | Normalized response verbosity relative to a reference length |
| `confidence_score` | Presence of assertive and declarative language |

---

## Key Findings (Initial Observations)

- Model responses are **highly sensitive to instruction phrasing** even when semantic meaning is preserved.
- **Adversarial prompts** significantly increase hallucination-like signals.
- **Overconstrained prompts** reduce uncertainty language but increase false confidence (instruction drift).
- **Context injection** creates non-linear changes in response structure.

---

## Implications for AI Safety

These findings suggest that:

- Model reliability is **not a static property** — it is context-dependent
- Evaluation must include **context perturbation testing**, not just accuracy benchmarks
- Safety frameworks should measure **stability**, not just accuracy

This aligns with active research directions in:
- scalable oversight
- robustness evaluation
- alignment under distribution shift

---

## Project Structure

```
ContextGuard/
├── README.md
├── requirements.txt
├── .env.example
│
├── app/
│   ├── config.py          # Env vars and model config
│   ├── llm_client.py      # OpenAI API wrapper
│   ├── prompts.py         # Prompt variant templates
│   ├── metrics.py         # Heuristic scoring functions
│   ├── evaluator.py       # Response evaluation and aggregation
│   └── utils.py           # JSON I/O and helpers
│
├── experiments/
│   ├── run_experiment.py  # Main experiment runner
│   ├── test_cases.json    # Question bank
│   └── prompt_sets.json   # Extended prompt variant library
│
├── dashboard/
│   └── streamlit_app.py   # Interactive results dashboard
│
├── results/
│   └── sample_results.json
│
└── data/
    └── (optional logs / exports)
```

---

## Tech Stack

- **Python 3.10+**
- **OpenAI API** (`openai` SDK)
- **Streamlit** — interactive dashboard
- **Pandas** — data manipulation
- **NumPy** — statistical analysis
- **Matplotlib** — visualization
- **python-dotenv** — environment management
- **tqdm** — progress bars

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ag1161/contextguard.git
cd contextguard
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run experiments

```bash
python experiments/run_experiment.py
```

Results are written to `results/sample_results.json`.

### 4. Launch dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## Sample Output

```
Running 5 questions × 4 variants = 20 evaluations

── Aggregate Results ──────────────────────────────────
  total_evaluations              20
  avg_hallucination_score        0.042
  avg_length_penalty             0.187
  avg_confidence_score           0.531
  avg_word_count                 37.4
────────────────────────────────────────────────────
```

---

## Future Work

- [ ] Embedding-based semantic consistency metrics
- [ ] LLM-as-judge evaluation layer for factual accuracy
- [ ] Multi-model comparison (GPT-4o vs Claude vs open models)
- [ ] Statistical significance testing across prompt sets (t-tests, effect sizes)
- [ ] Adversarial prompt injection suite
- [ ] Human evaluation annotation pipeline
- [ ] Publishable paper format with reproducibility checklist

---

## Research Positioning

ContextGuard is not a production application.

It is an exploratory research framework aimed at understanding model behavior under controlled perturbations.

The goal is to support:
- alignment research
- evaluation methodology design
- robustness analysis in LLM systems

This project is intended as a step toward empirical AI safety research — exploring how prompt sensitivity, hallucination risk, and instruction robustness can be measured and compared systematically.

---

## License

MIT
