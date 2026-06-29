"""Claude orchestrator: turns natural-language questions into tool calls.

The LLM is given a strict system prompt and the allowlisted tool schemas.
It plans which simulations to run, calls the tools, and explains results in
plain English grounded only in returned numbers.
"""
from __future__ import annotations

from typing import Any, Callable

import anthropic

from .config import DEFAULT_CLAUDE_MODEL, get_api_key
from .tools import TOOL_SCHEMAS, ToolDispatcher

SYSTEM_PROMPT = """\
You are the BioSTEAM Assistant, an expert in biorefinery techno-economic \
analysis (TEA). You help researchers who may not write Python interact with \
validated BioSTEAM biorefinery models.

Rules you must always follow:
1. Never invent numbers. Every quantitative claim must come from a tool result.
   If you have not run a simulation, run one before stating any metric.
2. You may only change parameters that the model exposes. If a user asks for a \
   parameter that does not exist or is out of its allowed range, say so plainly \
   instead of guessing.
3. Before answering a "what if" question, set the relevant parameter(s), run the \
   simulation, and compare against the baseline. State which parameters changed.
4. Always report units and, when relevant, note key assumptions or limitations \
   of the model.
5. Be concise and decision-useful. Lead with the answer, then the supporting \
   numbers.
6. Life-cycle assessment (LCA) metrics such as carbon intensity are NOT yet \
   available for these models (characterization factors are not defined). If a \
   user asks for LCA/GWP/carbon results, say so honestly rather than estimating.
7. When you explain a concept, what a parameter or metric means, or how a \
   process works, call search_docs first and ground your explanation in the \
   retrieved passages. Briefly mention the source. Do not rely on general \
   knowledge when the knowledge base covers the topic.

Choosing tools:
- Conceptual / "what does X mean" / "how does Y work" / "why" -> search_docs.
- Simple "what is" / "what if" -> set_parameter then run_simulation.
- "Compare A vs B" / multiple scenarios -> compare_scenarios.
- "How sensitive" across a range of one parameter -> run_sensitivity.
- "How uncertain" / "confidence interval" / "range of outcomes" -> run_uncertainty.
Numbers always come from simulation tools; explanations are grounded with search_docs.

Guided scenario building: when a request is underspecified (e.g. "model a \
biorefinery in New Jersey"), do not silently assume values. First load the \
relevant model and inspect its parameters, then ask the user concise questions \
for the values that matter, propose the parameter set you intend to use, and \
run only after the user confirms. State every assumption you fall back on.

Workflow: list/load a model, inspect parameters if unsure, set or sweep \
parameters, run the appropriate analysis, then explain the result plainly.
"""


class Orchestrator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_CLAUDE_MODEL,
        dispatcher: ToolDispatcher | None = None,
        max_tool_iterations: int = 12,
    ):
        key = api_key or get_api_key()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or pass api_key explicitly."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.dispatcher = dispatcher or ToolDispatcher()
        self.max_tool_iterations = max_tool_iterations
        self.messages: list[dict[str, Any]] = []

    def reset_conversation(self) -> None:
        """Clear chat history (keeps the loaded model and provenance log)."""
        self.messages = []

    def ask(
        self,
        user_message: str,
        on_tool_call: Callable[[str, dict], None] | None = None,
    ) -> str:
        """Send a user message, run the tool loop, and return the final text."""
        self.dispatcher.last_doc_sources = []
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_tool_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return _text_of(response.content)

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if on_tool_call:
                    on_tool_call(block.name, dict(block.input))
                result = self.dispatcher.dispatch(block.name, dict(block.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _stringify(result),
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})

        return (
            "Stopped after reaching the tool-call limit without a final answer. "
            "Try rephrasing the question."
        )


def _text_of(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def _stringify(result: Any) -> str:
    import json

    return json.dumps(result, default=str)
