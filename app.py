"""Streamlit chat UI for the BioSTEAM AI Assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from biosteam_ai.config import get_api_key
from biosteam_ai.orchestrator import Orchestrator

st.set_page_config(page_title="BioSTEAM Assistant", page_icon="🌱", layout="wide")
st.title("BioSTEAM Assistant")
st.caption("Ask biorefinery TEA/LCA questions in plain English. Answers are grounded in real BioSTEAM simulations.")


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
    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = []


_init_state()

with st.sidebar:
    st.header("Status")
    if get_api_key():
        st.success("Anthropic API key detected")
    else:
        st.error("No ANTHROPIC_API_KEY. Copy .env.example to .env and add your key.")

    orch = st.session_state.orchestrator
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
        "- Load the corn stover model and tell me the minimum ethanol selling price.\n"
        "- What happens to ethanol price if fermentation efficiency drops to 80%?\n"
        "- Run a sensitivity analysis on feedstock price from 0.04 to 0.08."
    )

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

            try:
                answer = st.session_state.orchestrator.ask(prompt, on_tool_call=_on_tool_call)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                answer = f"Error: {exc}"
                status.update(label="Error", state="error")

            st.markdown(answer)
            st.session_state.history.append({"role": "assistant", "content": answer})
