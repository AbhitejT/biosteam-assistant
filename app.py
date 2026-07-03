"""Streamlit chat UI for the BioSTEAM AI Assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import biosteam

from biosteam_ai.config import get_api_key
from biosteam_ai.orchestrator import Orchestrator
from biosteam_ai.reporting import build_json_report, build_markdown_report

st.set_page_config(page_title="BioSTEAM Assistant", page_icon="🌱", layout="wide")
st.title("BioSTEAM Assistant")
st.caption("Ask biorefinery techno-economic questions in plain English. Answers are grounded in real BioSTEAM simulations.")

PRIMARY_METRIC = "mesp_per_gal"


def _primary(metrics_keys) -> str:
    return PRIMARY_METRIC if PRIMARY_METRIC in metrics_keys else next(iter(metrics_keys))


def render_artifact(artifact: dict) -> None:
    """Render a chartable result produced by the simulation engine."""
    kind = artifact["kind"]
    data = artifact["data"]
    units = data.get("metric_units", {})

    if kind == "sensitivity":
        rows = data["sweep"]
        if not rows:
            return
        metric = _primary(rows[0]["metrics"])
        df = pd.DataFrame(
            {
                data["parameter"]: [r["parameter_value"] for r in rows],
                metric: [r["metrics"][metric] for r in rows],
            }
        ).set_index(data["parameter"])
        st.markdown(f"**Sensitivity: {metric} ({units.get(metric, '')}) vs {data['parameter']}**")
        st.line_chart(df)

    elif kind == "comparison":
        base = data["baseline"]
        scen = data["scenarios"]
        metric = _primary(base)
        names = ["baseline"] + list(scen)
        vals = [base[metric]] + [scen[n]["metrics"][metric] for n in scen]
        df = pd.DataFrame({"scenario": names, metric: vals}).set_index("scenario")
        st.markdown(f"**Scenario comparison: {metric} ({units.get(metric, '')})**")
        st.bar_chart(df)

    elif kind == "uncertainty":
        samples = data["samples"]
        metric = _primary(samples)
        vals = samples[metric]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(vals, bins=20, color="#4c9a6a", edgecolor="white")
        ax.set_xlabel(f"{metric} ({units.get(metric, '')})")
        ax.set_ylabel("frequency")
        ax.set_title(f"Monte Carlo ({data['n_samples']} samples)")
        st.markdown(f"**Uncertainty distribution: {metric}**")
        st.pyplot(fig)


def render_verification(report: dict) -> None:
    """Render a model validation report card."""
    overall = report["overall"]
    label = {"pass": "PASS", "warn": "REVIEW RECOMMENDED", "fail": "FAIL"}[overall]
    banner = {"pass": st.success, "warn": st.warning, "fail": st.error}[overall]
    icon = {"pass": "✓", "warn": "!", "fail": "✕"}
    banner(f"Model validation: {label}")
    with st.expander("Validation checks", expanded=(overall != "pass")):
        for c in report["checks"]:
            st.markdown(f"{icon[c['status']]} **{c['name']}** — {c['detail']}")


def render_built_process(built: dict) -> None:
    """Render a from-scratch process the assistant assembled and simulated."""
    results = built["results"]
    st.markdown(f"**Built process: {results['name']}**")

    stream_rows = []
    for group, role in (("feeds", "feed"), ("intermediates", "intermediate"),
                        ("products", "product")):
        for s in results.get(group, []):
            stream_rows.append({
                "stream": s["name"], "role": role, "phase": s["phase"],
                "T (K)": s["T_K"], "kg/hr": s["F_mass_kg_hr"],
                "kmol/hr": s["F_mol_kmol_hr"],
            })
    if stream_rows:
        st.caption("Streams")
        st.dataframe(pd.DataFrame(stream_rows), hide_index=True)

    unit_rows = [
        {"unit": u["id"], "type": u["type"],
         "installed cost (USD)": u["installed_cost_usd"]}
        for u in results.get("units", [])
    ]
    if unit_rows:
        st.caption("Unit operations")
        st.dataframe(pd.DataFrame(unit_rows), hide_index=True)
    st.caption(
        f"Total installed equipment cost: "
        f"${results['total_installed_equipment_cost_usd']:,.0f}"
    )

    econ = results.get("economics")
    if econ:
        msp = econ["min_selling_price_usd_per_kg"]
        c1, c2 = st.columns(2)
        c1.metric(
            f"Min. selling price ({econ['product']})", f"${msp:,.3f}/kg"
        )
        c2.metric(
            "Total capital investment",
            f"${econ['total_capital_investment_usd']:,.0f}",
        )
        a = econ["assumptions"]
        st.caption(
            "Simplified TEA estimate — assumptions: "
            f"IRR {a['IRR']:.0%}, {int(a['plant_years'])} yr, "
            f"{int(a['operating_days'])} days/yr, Lang {a['lang_factor']}, "
            f"FOC {a['FOC_over_FCI']:.0%} of FCI. "
            f"Material cost ${econ['material_cost_usd_per_yr']:,.0f}/yr."
        )

    carbon = results.get("carbon")
    if carbon:
        ghg = carbon.get("direct_ghg_kg_per_hr") or {}
        if ghg:
            intensity = carbon.get("co2e_kg_per_kg_product")
            label = "Direct CO2e"
            value = f"{carbon['co2e_kg_per_hr']:,.0f} kg/hr"
            if intensity is not None:
                value = f"{intensity:,.3f} kg/kg product"
                label = "Direct carbon intensity"
            st.metric(label, value)
            st.caption(
                f"Direct process outlet emissions ({', '.join(ghg)}), "
                f"{carbon['gwp_basis']}. Gate-to-gate direct only — not a full "
                "cradle-to-grave LCA."
            )
        else:
            st.caption("No direct greenhouse gases in outlet streams.")

    if built.get("verification"):
        render_verification(built["verification"])


def render_sources(sources: list[dict]) -> None:
    """Show which knowledge-base passages grounded the explanation."""
    seen = set()
    unique = []
    for s in sources:
        key = (s["title"], s["source"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    with st.expander(f"Sources ({len(unique)} knowledge-base passages)", expanded=False):
        for s in unique:
            st.caption(f"- **{s['title']}** — _{s['source']}_ (relevance {s['score']})")


def _init_state() -> None:
    if "orchestrator" not in st.session_state:
        try:
            st.session_state.orchestrator = Orchestrator()
            st.session_state.error = None
        except Exception as exc:  # missing API key, etc.
            st.session_state.orchestrator = None
            st.session_state.error = str(exc)
    if "history" not in st.session_state:
        st.session_state.history = []


_init_state()

with st.sidebar:
    st.header("Status")
    if get_api_key():
        st.success("Anthropic API key detected")
    else:
        st.error("No ANTHROPIC_API_KEY. Copy .env.example to .env and add your key.")

    orch = st.session_state.orchestrator
    if orch:
        st.caption(f"Model: `{orch.model}`")

    if st.button("New conversation"):
        st.session_state.history = []
        if orch:
            orch.reset_conversation()
        st.rerun()

    if orch and orch.dispatcher.session.is_loaded:
        spec = orch.dispatcher.session.spec
        st.subheader("Loaded model")
        st.write(f"**{spec.name}** (`{spec.key}`)")
        with st.expander("Current parameters", expanded=False):
            for p in orch.dispatcher.session.list_parameters():
                st.write(
                    f"- **{p['name']}** = {p['current_value']:.4g} {p['units']} "
                    f"(bounds {p['bounds'][0]}–{p['bounds'][1]})"
                )

    if orch:
        st.subheader("Provenance log")
        st.caption(f"`{orch.dispatcher.log_path.name}`")

    if orch and st.session_state.history:
        st.subheader("Export report")
        log_name = orch.dispatcher.log_path.name
        md = build_markdown_report(
            st.session_state.history, orch.model, log_name, biosteam.__version__
        )
        js = build_json_report(
            st.session_state.history, orch.model, log_name, biosteam.__version__
        )
        st.download_button(
            "Download report (Markdown)", md,
            file_name="biosteam_session_report.md", mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "Download data (JSON)", js,
            file_name="biosteam_session_report.json", mime="application/json",
            use_container_width=True,
        )

    st.subheader("Try asking")
    st.markdown(
        "- What models can I use?\n"
        "- Load corn stover and give me the minimum ethanol selling price.\n"
        "- Compare ethanol price at 80% vs 95% fermentation conversion.\n"
        "- Run a sensitivity analysis on feedstock price from 0.04 to 0.08.\n"
        "- How uncertain is the price if conversion ranges 0.80-0.97?\n"
        "- Build a process: react ethanol + acetic acid to ethyl acetate, then flash it.\n"
        "- Add a recycle loop to recover unreacted feed and recompute the price.\n"
        "- What are the direct CO2 emissions of that process?"
    )

if not st.session_state.history:
    with st.container(border=True):
        st.markdown(
            "#### Welcome\n"
            "This assistant runs **real BioSTEAM simulations** for you — no coding needed. "
            "Every number comes from a simulation, and each result can be validated and exported.\n\n"
            "**You can:**\n"
            "- Analyze curated biorefinery models (corn stover, sugarcane, lipidcane): "
            "prices, sensitivity, scenario comparison, Monte Carlo uncertainty.\n"
            "- **Build a new process from scratch** — feeds, reactor, flash, splitter, mixer, "
            "heater, and recycle loops — then get its equipment cost, minimum selling price, "
            "and direct carbon emissions.\n"
            "- Ask *why*/*what does this mean* and get explanations grounded in a knowledge base.\n\n"
            "Try one of the examples in the sidebar to get started."
        )

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("artifact"):
            render_artifact(msg["artifact"])
        if msg.get("verification"):
            render_verification(msg["verification"])
        if msg.get("built"):
            render_built_process(msg["built"])
        if msg.get("sources"):
            render_sources(msg["sources"])

prompt = st.chat_input("Ask about a biorefinery model...")
if prompt:
    if st.session_state.orchestrator is None:
        st.error(st.session_state.error or "Assistant unavailable.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Thinking and running simulations...", expanded=True)

            def _on_tool_call(name: str, args: dict) -> None:
                status.write(f"`{name}` {args if args else ''}")

            orch = st.session_state.orchestrator
            try:
                answer = orch.ask(prompt, on_tool_call=_on_tool_call)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                answer = f"Error: {exc}"
                status.update(label="Error", state="error")

            st.markdown(answer)
            artifact = None
            if orch.dispatcher.session.is_loaded and orch.dispatcher.session.last_artifact:
                artifact = copy.deepcopy(orch.dispatcher.session.last_artifact)
                render_artifact(artifact)
                orch.dispatcher.session.last_artifact = None

            verification = None
            if orch.dispatcher.session.is_loaded and orch.dispatcher.session.last_verification:
                verification = copy.deepcopy(orch.dispatcher.session.last_verification)
                render_verification(verification)
                orch.dispatcher.session.last_verification = None

            built = None
            if orch.dispatcher.builder.last_results:
                built = {
                    "results": copy.deepcopy(orch.dispatcher.builder.last_results),
                    "verification": copy.deepcopy(
                        orch.dispatcher.builder.last_verification
                    ),
                }
                render_built_process(built)
                orch.dispatcher.builder.last_results = None
                orch.dispatcher.builder.last_verification = None

            sources = list(orch.dispatcher.last_doc_sources)
            if sources:
                render_sources(sources)

            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "artifact": artifact,
                    "verification": verification,
                    "built": built,
                    "sources": sources,
                }
            )
