# BioSTEAM AI Assistant

A validated natural-language interface over a curated set of
[BioSTEAM](https://biosteam.readthedocs.io/) biorefinery models. Ask
techno-economic questions in plain English; every answer is grounded in a real
BioSTEAM simulation, not in the language model's imagination.

Phase 1 delivered a **validated query assistant** over pre-built models, with
numerical regression tests and full provenance logging. Phase 2 added **scenario
comparison, Monte Carlo uncertainty analysis, guided multi-turn scenario
building, RAG-grounded explanations, and charts** in the web UI. Phase 3 added a
**guarded process builder**: the assistant can assemble, simulate, and verify a
*new* flowsheet from typed building blocks — not just drive pre-wired models.
Phase 4 rounds it out for real use: **recycle loops**, a **minimum selling
price** and **direct carbon emissions** on built processes, and **downloadable
session reports**.

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
| `verify_model` | correctness checks (mass balance, reactions, plausibility) with pass/warn/fail |
| `list_building_blocks` | list the chemicals + unit blocks available for building new processes |
| `build_process` | assemble, simulate, and verify a **new** flowsheet from typed blocks (with recycles, selling price, and direct carbon) |
| `search_docs` | retrieve passages from the knowledge base to ground explanations (RAG) |

Comparison, sensitivity, and uncertainty results are rendered as charts in the
Streamlit UI; grounded explanations show their knowledge-base sources; and
verification produces a model-validation card.

### Verification layer (trust, not just "it ran")

A simulation that converges can still be wrong. `verify_model` checks that the
*current* model is physically and economically sane and returns a report with an
overall status of **pass**, **warn** (review recommended), or **fail**:

- **Per-unit mass balance** — every unit's inputs must equal its outputs (warning).
- **No negative flows** — no negative component flows anywhere (error).
- **Reaction mass balance** — reaction stoichiometry conserves mass (warning).
- **Output plausibility** — metrics finite; MESP and production in sane ranges (error).

The assistant is instructed to surface any warning/failure and temper its answer
accordingly, rather than presenting flagged numbers as fully trustworthy. These
checks live in `biosteam_ai/verification.py` and are shared verbatim by both the
curated models and the process builder, so "verified" means the same thing
everywhere.

### Process builder (assembling new flowsheets)

Beyond driving curated models, the assistant can build a *new* process from a
declarative spec: a set of chemicals, feed streams, and unit blocks wired
together by stream name. `build_process` validates the spec, constructs a real
BioSTEAM flowsheet, simulates it, and runs the verification layer above — all in
one guarded step. Supported blocks (see `list_building_blocks`):

| Block | What it does |
|-------|--------------|
| `mixer` | combine 2+ inlet streams into one |
| `reactor` | one stoichiometric reaction at a fixed conversion (mass conserved exactly) |
| `flash` | vapour/liquid split by equilibrium at a given vapour fraction |
| `splitter` | split one stream into two by a fixed fraction |
| `heater` | heat/cool a stream to a target temperature (with duty + cost) |

Safety comes from **allowlists**: only a curated set of chemicals (validated to
have complete thermo data and survive a flash) and the block types above are
permitted; anything else is rejected with a clear message. The custom reactor
block conserves mass and never crashes on costing, unlike BioSTEAM's stirred-tank
reactor classes on gas-phase or isothermal duties.

**Recycle loops.** Real biorefineries recycle unreacted feed. Declare recycle
(tear) stream names in `recycles` and they may be read before they are produced;
the builder creates them as empty tear streams and lets the solver converge the
loop. Each recycle must be produced by exactly one unit and consumed by another
(validated up front). Recovering unreacted feed through a recycle typically
lowers the minimum selling price.

**Economics (minimum selling price).** Beyond equipment cost, a built process
can return a **minimum product selling price**. Give each feed a `price`
(USD/kg) and name the terminal `product` stream to price; the builder wraps the
flowsheet in a `SimpleTEA` (Lang-factor capital from installed equipment cost,
fixed operating cost as a fraction of fixed capital) and solves the break-even
price. Financial assumptions default to IRR 10%, 20-year life, 330 operating
days/yr, Lang factor 3.0, and FOC = 5% of FCI, and can be overridden per build
via `economics`. An **economic-plausibility** check (price finite and positive)
is added to the verification report.

**Direct carbon emissions.** Results include a `carbon` section: the direct
greenhouse gases (CO2, CH4) leaving the process in outlet streams, expressed as
CO2-equivalent using IPCC AR5 GWP100 factors, plus an intensity per kg of
product. This is a **gate-to-gate direct** figure — it deliberately excludes
upstream feedstock/energy burdens and biogenic-carbon accounting, so it is not a
full cradle-to-grave LCA, and the assistant states that scope.

**Honest boundaries.** The built-process selling price comes from a *simplified*
TEA and is an estimate — the curated registry models, with their bespoke TEA
wiring, remain the reference for fully-validated prices; and the carbon figure is
direct emissions only, not a full LCA. The assistant is told to state its
assumptions and point to the registry models when a request exceeds the palette.

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

### Exportable reports

Any session can be exported from the sidebar as a **Markdown report** (a
readable record of the conversation plus built-process streams, economics,
carbon, and validation) or as **JSON** (the full structured results). Both embed
the assistant model and BioSTEAM version, and reference the provenance log, so an
analysis is reproducible and shareable without touching code.

### Not yet supported (deferred)

- **Full life-cycle assessment (cradle-to-grave GWP).** Built processes report
  *direct* process CO2e (see above), but a complete LCA — upstream feedstock and
  energy burdens, biogenic-carbon credits, allocation across co-products — is not
  included. The curated `cornstover`/`sugarcane`/`lipidcane` models also do not
  define life-cycle characterization factors, so the assistant reports direct
  emissions honestly rather than fabricating a full footprint.

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

Web UI (quickstart — checks your `.env`, then launches):

```bash
./run.sh
```

or directly:

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
- "Build a process: react 100 kmol/hr ethanol with 100 kmol/hr acetic acid to ethyl
  acetate at 60% conversion, then flash it. Is it valid?"
- "...and if the feed costs $0.50/kg, what's the minimum selling price of the product?"
- "Add a recycle loop that sends 40% of the liquid back to the reactor and recompute the price."
- "What are the direct CO2-equivalent emissions of that process?"

## Tests

The regression suite runs without an API key (it exercises the simulation layer
directly and compares against raw BioSTEAM):

```bash
pytest -v
```

## Project layout

```
biosteam_ai/
  config.py             # env + constants
  models/registry.py    # declarative model specs (params, metrics, bounds)
  engine.py             # SimulationSession: load/set/run/reset/sensitivity/compare/uncertainty/verify
  verification.py       # shared correctness checks (used by engine + builder)
  builder/blocks.py     # chemical allowlist + typed unit-block palette
  builder/process_builder.py  # ProcessBuilder + SimpleTEA: spec -> construct -> simulate -> price -> carbon -> verify
  reporting.py          # Markdown/JSON session report generation
  tools.py              # LLM tool schemas + logging dispatcher
  orchestrator.py       # Claude tool-calling loop
  rag/retriever.py      # TF-IDF retrieval over the knowledge base
  rag/knowledge/*.md    # curated BioSTEAM/TEA knowledge
  cli.py                # terminal chat
app.py                  # Streamlit chat UI (charts, sources, built processes, report export)
run.sh                  # quickstart launcher for the web UI
tests/test_engine.py    # regression tests vs raw BioSTEAM
tests/test_builder.py   # process-builder tests (assemble, recycle, economics, carbon, verify)
tests/test_rag.py       # retrieval tests
```

## What's next

- Extend the process builder further: more unit types (distillation columns,
  pumps/compressors) and a larger chemical palette.
- Full cradle-to-grave LCA using literature-sourced characterization factors.
- Upgrade retrieval from TF-IDF to embeddings, and expand the knowledge corpus.
- Deploy the web app publicly and run a non-programmer user study.
