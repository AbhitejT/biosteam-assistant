"""Streamlit chat UI for the BioSTEAM AI Assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from biosteam_ai.config import get_api_key
from biosteam_ai.orchestrator import Orchestrator

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

    st.subheader("Try asking")
    st.markdown(
        "- What models can I use?\n"
        "- Load corn stover and give me the minimum ethanol selling price.\n"
        "- Compare ethanol price at 80% vs 95% fermentation conversion.\n"
        "- Run a sensitivity analysis on feedstock price from 0.04 to 0.08.\n"
        "- How uncertain is the price if conversion ranges 0.80-0.97?"
    )

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("artifact"):
            render_artifact(msg["artifact"])
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

            sources = list(orch.dispatcher.last_doc_sources)
            if sources:
                render_sources(sources)

            st.session_state.history.append(
                {"role": "assistant", "content": answer, "artifact": artifact, "sources": sources}
            )
