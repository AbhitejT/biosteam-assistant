# BioSTEAM AI Assistant (Phase 1 MVP)

A validated natural-language interface over a curated set of
[BioSTEAM](https://biosteam.readthedocs.io/) biorefinery models. Ask
techno-economic questions in plain English; every answer is grounded in a real
BioSTEAM simulation, not in the language model's imagination.

This is the Phase 1 deliverable from `biosteam_ai_assistant_plan.txt`: a
**validated query assistant** over pre-built models, with numerical regression
tests and full provenance logging.

## Why this design

The assistant does **not** let the LLM write free-form simulation code. Instead
the LLM drives a typed, allowlisted tool layer:

```
User -> Claude orchestrator -> typed tools -> BioSTEAM models -> TEA metrics
                                   |
                                   +-> provenance log (every call + result)
```

- Parameters are **bounded and validated** (out-of-range or unknown params are rejected).
- Every tool call and result is **logged** to `run_logs/` with model + BioSTEAM version.
- Outputs are checked against raw BioSTEAM in `tests/` so results stay correct.

## Models included

| Key | Model |
|-----|-------|
| `cornstover` | Corn stover cellulosic ethanol (NREL biochemical design) |
| `sugarcane`  | Sugarcane first-generation ethanol |

Adjustable parameters per model include feedstock price, fermentation
conversion/efficiency, plant lifetime, and target IRR. Metrics include minimum
ethanol selling price (MESP, USD/gal and USD/kg), ethanol production rate, and
total capital investment.

## Setup

Requires **Python 3.12+** (current BioSTEAM/thermosteam use 3.12 syntax).

```bash
conda create -n biosteam-ai python=3.12 -y
conda activate biosteam-ai
pip install -r requirements.txt
```

Add your Anthropic API key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...
```

## Run

Web UI:

```bash
streamlit run app.py
```

Terminal chat:

```bash
python -m biosteam_ai.cli
```

## Example questions

- "What models can I use?"
- "Load the corn stover model and tell me the minimum ethanol selling price."
- "What happens to ethanol price if fermentation efficiency drops to 80%?"
- "Run a sensitivity analysis on feedstock price from 0.04 to 0.08 USD/kg."

## Tests

The regression suite runs without an API key (it exercises the simulation layer
directly and compares against raw BioSTEAM):

```bash
pytest -v
```

## Project layout

```
biosteam_ai/
  config.py           # env + constants
  models/registry.py  # declarative model specs (params, metrics, bounds)
  engine.py           # SimulationSession: load/set/run/reset/sensitivity
  tools.py            # LLM tool schemas + logging dispatcher
  orchestrator.py     # Claude tool-calling loop
  cli.py              # terminal chat
app.py                # Streamlit chat UI
tests/test_engine.py  # regression tests vs raw BioSTEAM
```

## What's next (Phases 2-3)

- Multi-turn guided scenario configuration and richer LCA metrics.
- RAG over BioSTEAM docs for explanation grounding.
- Deployed web app, downloadable reports, and a non-programmer user study.
