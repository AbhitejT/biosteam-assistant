# BioSTEAM AI Assistant (Phase 1 MVP)

A validated natural-language interface over a curated set of
[BioSTEAM](https://biosteam.readthedocs.io/) biorefinery models. Ask
techno-economic questions in plain English; every answer is grounded in a real
BioSTEAM simulation, not in the language model's imagination.

Phase 1 delivered a **validated query assistant** over pre-built models, with
numerical regression tests and full provenance logging. Phase 2 adds **scenario
comparison, Monte Carlo uncertainty analysis, guided multi-turn scenario
building, and charts** in the web UI.

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
| `lipidcane`  | Lipidcane co-producing ethanol + biodiesel |

> Note: the `biorefineries` package contains more models, but many do not load
> with the currently pinned BioSTEAM version (API drift) or lack a clean
> single-product TEA. Models are added only after verifying they load, run, and
> respond correctly to parameter changes.

Adjustable parameters per model include feedstock price, fermentation
conversion/efficiency, plant lifetime, and target IRR. Metrics include minimum
ethanol selling price (MESP, USD/gal and USD/kg), ethanol production rate, and
total capital investment.

## Capabilities

The LLM drives an allowlisted set of tools:

| Tool | Use |
|------|-----|
| `list_models` / `load_model` | discover and load a model |
| `list_parameters` / `set_parameter` / `reset_parameters` | inspect and adjust bounded parameters |
| `run_simulation` | re-simulate and read all metrics |
| `compare_scenarios` | run named scenarios and report deltas vs. baseline |
| `run_sensitivity` | sweep one parameter across a range |
| `run_uncertainty` | Monte Carlo over parameter distributions (mean, std, p5/p50/p95) |
| `search_docs` | retrieve passages from the knowledge base to ground explanations (RAG) |

Comparison, sensitivity, and uncertainty results are rendered as charts in the
Streamlit UI, and grounded explanations show their knowledge-base sources.

### Retrieval-augmented explanations (RAG)

Quantitative answers always come from simulation; *explanations* are grounded in
a curated knowledge base via the `search_docs` tool. The corpus combines:

- hand-written passages under `biosteam_ai/rag/knowledge/` (BioSTEAM overview,
  techno-economic glossary, and process background), and
- auto-generated documentation of the live model registry (every model,
  parameter, and metric), so descriptions never drift from the code.

Retrieval uses TF-IDF + cosine similarity (scikit-learn) — no embedding API or
extra credentials needed. The `Retriever` interface (`search(query, k)`) is
designed so an embedding/vector backend can replace it without changing callers.

### Not yet supported (deferred)

- **LCA / carbon intensity (GWP).** The bundled `cornstover` and `sugarcane`
  models do not define life-cycle characterization factors, so the assistant
  will not report carbon metrics rather than fabricate them. Adding LCA requires
  sourcing characterization factors (e.g. from GREET) — planned for a later
  increment.

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
- "Compare ethanol price at 80% vs 95% fermentation conversion."
- "Run a sensitivity analysis on feedstock price from 0.04 to 0.08 USD/kg."
- "How uncertain is the ethanol price if conversion ranges from 0.80 to 0.97?"

## Tests

The regression suite runs without an API key (it exercises the simulation layer
directly and compares against raw BioSTEAM):

```bash
pytest -v
```

## Project layout

```
biosteam_ai/
  config.py            # env + constants
  models/registry.py   # declarative model specs (params, metrics, bounds)
  engine.py            # SimulationSession: load/set/run/reset/sensitivity/compare/uncertainty
  tools.py             # LLM tool schemas + logging dispatcher
  orchestrator.py      # Claude tool-calling loop
  rag/retriever.py     # TF-IDF retrieval over the knowledge base
  rag/knowledge/*.md   # curated BioSTEAM/TEA knowledge
  cli.py               # terminal chat
app.py                 # Streamlit chat UI (charts + sources)
tests/test_engine.py   # regression tests vs raw BioSTEAM
tests/test_rag.py      # retrieval tests
```

## What's next

- LCA metrics (carbon intensity) using literature-sourced characterization factors.
- Upgrade retrieval from TF-IDF to embeddings, and expand the knowledge corpus.
- Deployed web app, downloadable reports, and a non-programmer user study.
