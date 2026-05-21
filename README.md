# ContextGuard

> An experimental framework for evaluating how large language models respond under different prompt structures and contextual conditions.

---

## Motivation

Modern LLMs are increasingly deployed in high-stakes environments — finance, healthcare, legal compliance — yet their behavior is highly sensitive to prompt framing and context injection. A subtly different prompt can produce a confidently wrong answer where a well-formed one produces the correct one.

ContextGuard is a controlled evaluation harness designed to surface early empirical signals of:

- **Hallucination sensitivity** — does hedging language increase with prompt adversariality?
- **Instruction hierarchy robustness** — can simple adversarial injections shift model behavior?
- **Context-driven variability** — how much does response structure change across prompt types?
- **Response stability** — are outputs meaningfully different across variants for factual questions?

---

## Key Research Questions

1. How does prompt structure affect model reliability on the same underlying question?
2. Do adversarial prompts increase hedging and hallucination signals?
3. How stable are model outputs across semantic-preserving context variations?
4. Can lightweight heuristic metrics serve as proxies before expensive human evaluation?

---

## Methodology

Experiments follow a **2D grid design**:

```
Questions × Prompt Variants → LLM responses → Scored metrics
```

### Prompt Variants

| Variant | Description |
|---|---|
| `baseline` | Neutral, clean question framing |
| `adversarial` | Injection-style override attempt |
| `ambiguous` | Underspecified, probes interpretation |
| `overconstrained` | Rigid format constraint, tests compliance |

### Metrics (heuristic proxies)

| Metric | Description |
|---|---|
| `hallucination_score` | Fraction of hedging-language signals detected |
| `length_penalty` | Normalised response verbosity |
| `confidence_score` | Presence of assertive/declarative language |

These are fast, deterministic first-pass signals. They are not ground-truth measures; they are designed to be interpretable and reproducible as a baseline before integrating LLM-as-judge or human evaluation.

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
- **Matplotlib** — visualisation
- **python-dotenv** — environment management
- **tqdm** — progress bars

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/ContextGuard.git
cd ContextGuard
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

- [ ] **Embedding-based hallucination detection** — compare response embeddings to ground-truth vectors
- [ ] **LLM-as-judge evaluation** — use a secondary model to score factual accuracy
- [ ] **Multi-model benchmarking** — compare GPT-4o, Claude, Gemini under identical conditions
- [ ] **Statistical significance testing** — t-tests and effect sizes across variants
- [ ] **Human evaluation pipeline** — annotation interface for gold-standard labelling
- [ ] **Adversarial prompt injection suite** — systematic exploration of jailbreak surfaces
- [ ] **Publishable paper format** — structured experimental write-up with reproducibility checklist

---

## Research Direction

ContextGuard is a stepping stone toward empirical evaluation of model reliability in real-world decision systems. The core hypothesis is that prompt sensitivity is a measurable, quantifiable property — and that understanding it is a prerequisite for deploying LLMs responsibly in high-stakes contexts.

---

## License

MIT
