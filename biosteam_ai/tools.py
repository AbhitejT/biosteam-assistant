"""Tool definitions exposed to the LLM, plus a logging dispatcher.

The LLM may only call these allowlisted tools. Every call and result is
appended to a provenance log so any AI-assisted answer is reproducible.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import biosteam

from .builder import ProcessBuilder, palette
from .config import RUN_LOG_DIR
from .engine import SimulationSession, available_models

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_models",
        "description": "List the curated BioSTEAM biorefinery models available to load.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "load_model",
        "description": (
            "Load a biorefinery model by key (e.g. 'cornstover' or 'sugarcane'). "
            "Must be called before reading or changing parameters. Returns the "
            "model's adjustable parameters and their current baseline values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_key": {"type": "string", "description": "Model key to load."}
            },
            "required": ["model_key"],
        },
    },
    {
        "name": "list_parameters",
        "description": "List adjustable parameters for the loaded model, with units, bounds, and current values.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_parameter",
        "description": (
            "Set one adjustable parameter on the loaded model. The value must lie "
            "within the parameter's allowed bounds, otherwise the call is rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["name", "value"],
        },
    },
    {
        "name": "reset_parameters",
        "description": "Reset all parameters of the loaded model back to their baseline values.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_simulation",
        "description": (
            "Re-simulate the loaded model with the current parameters and return "
            "all techno-economic metrics (including minimum ethanol selling price)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_sensitivity",
        "description": (
            "Sweep one parameter across a list of values, re-simulating at each "
            "point, and return the resulting metrics. Parameters are restored to "
            "their prior value afterward."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parameter": {"type": "string"},
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Values to sweep the parameter across.",
                },
            },
            "required": ["parameter", "values"],
        },
    },
    {
        "name": "compare_scenarios",
        "description": (
            "Run one or more named scenarios and compare them against the "
            "baseline. Each scenario is a set of parameter overrides. Returns "
            "metrics, absolute deltas, and percent changes vs. baseline for each "
            "scenario. Use this for 'compare A vs B' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "object",
                    "description": (
                        "Mapping of scenario name -> object of {parameter: value}. "
                        "Example: {\"high enzyme\": {\"glucose_to_ethanol_conversion\": 0.97}}."
                    ),
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                }
            },
            "required": ["scenarios"],
        },
    },
    {
        "name": "run_uncertainty",
        "description": (
            "Run a Monte Carlo uncertainty analysis: draw the given parameter(s) "
            "from distributions over their range and report summary statistics "
            "(mean, std, and 5th/50th/95th percentiles) for every metric. Use this "
            "for 'how uncertain', 'confidence interval', or 'range of outcomes' "
            "questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "distributions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "low": {"type": "number"},
                            "high": {"type": "number"},
                            "kind": {
                                "type": "string",
                                "enum": ["uniform", "triangular"],
                            },
                            "mode": {"type": "number"},
                        },
                        "required": ["name", "low", "high"],
                    },
                },
                "n_samples": {
                    "type": "integer",
                    "description": "Number of Monte Carlo samples (default 100, max 1000).",
                },
            },
            "required": ["distributions"],
        },
    },
    {
        "name": "verify_model",
        "description": (
            "Run correctness checks on the currently loaded/simulated model and "
            "return a validation report: per-unit mass balance closure, absence of "
            "negative flows, reaction mass balance, and output plausibility. Use "
            "this when the user asks whether results are trustworthy/valid, or "
            "proactively before presenting results for an unfamiliar configuration. "
            "Overall status is pass, warn (review recommended), or fail."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_building_blocks",
        "description": (
            "List the palette for building a NEW process from scratch: the "
            "allowed chemicals and the unit-block types (mixer, reactor, flash, "
            "splitter, heater) with their parameters. Call this before "
            "build_process so you only use supported chemicals and blocks."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "build_process",
        "description": (
            "Assemble, simulate, and verify a NEW custom process from typed "
            "building blocks. Use this when the user wants to model a process "
            "that is not one of the curated models (e.g. 'build a reactor that "
            "converts X to Y then flash it'). Streams are referenced by name: a "
            "unit's 'ins' must be a feed name or an earlier unit's output; its "
            "'outs' name new streams. Flows are kmol/hr, temperatures K, "
            "pressures Pa. Returns stream results, equipment cost, direct "
            "greenhouse-gas emissions, and a verification report. To also get a "
            "minimum product selling price, give feeds a 'price' (USD/kg) and "
            "set 'product' to the terminal output stream to price; optionally "
            "override 'economics'. For recycle loops (feeding a downstream "
            "stream back upstream), list those stream names in 'recycles' so "
            "they may be read before they are produced. Only chemicals/blocks "
            "from list_building_blocks are allowed; invalid specs are rejected "
            "with a message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A label for the process."},
                "chemicals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chemicals present (must be in the allowlist).",
                },
                "recycles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names of recycle (tear) streams that are fed back "
                        "upstream. Each must be produced by exactly one unit and "
                        "consumed by another. Enables loops."
                    ),
                },
                "feeds": {
                    "type": "array",
                    "description": "Feed streams entering the process.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "flows": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                                "description": "Chemical -> molar flow (kmol/hr).",
                            },
                            "T": {"type": "number", "description": "Temperature K (default 298.15)."},
                            "P": {"type": "number", "description": "Pressure Pa (default 101325)."},
                            "price": {"type": "number", "description": "Feed price USD/kg (for TEA)."},
                        },
                        "required": ["name", "flows"],
                    },
                },
                "units": {
                    "type": "array",
                    "description": (
                        "Unit blocks in process order. Each has id, type, ins, "
                        "outs plus type-specific parameters: reactor needs "
                        "reaction/reactant/conversion (optional T,P); flash needs "
                        "V (optional P); splitter needs split; heater needs T."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["mixer", "reactor", "flash", "splitter", "heater"],
                            },
                            "ins": {"type": "array", "items": {"type": "string"}},
                            "outs": {"type": "array", "items": {"type": "string"}},
                            "reaction": {"type": "string"},
                            "reactant": {"type": "string"},
                            "conversion": {"type": "number"},
                            "V": {"type": "number"},
                            "split": {"type": "number"},
                            "T": {"type": "number"},
                            "P": {"type": "number"},
                        },
                        "required": ["id", "type", "ins", "outs"],
                    },
                },
                "product": {
                    "type": "string",
                    "description": (
                        "Name of the terminal output stream to solve a minimum "
                        "selling price for (optional; requires feed prices)."
                    ),
                },
                "economics": {
                    "type": "object",
                    "description": (
                        "Optional financial-assumption overrides. Defaults: "
                        "IRR 0.10, plant_years 20, income_tax 0.21, "
                        "operating_days 330, lang_factor 3.0, FOC_over_FCI 0.05."
                    ),
                    "properties": {
                        "IRR": {"type": "number"},
                        "plant_years": {"type": "number"},
                        "income_tax": {"type": "number"},
                        "operating_days": {"type": "number"},
                        "lang_factor": {"type": "number"},
                        "FOC_over_FCI": {"type": "number"},
                    },
                },
            },
            "required": ["chemicals", "feeds", "units"],
        },
    },
    {
        "name": "search_docs",
        "description": (
            "Search the curated BioSTEAM knowledge base (process background, "
            "techno-economic glossary, and model/parameter documentation) for "
            "explanatory text. Use this to ground explanations of concepts, what a "
            "parameter or metric means, or how a process works, instead of relying "
            "on general knowledge. Returns the most relevant passages with their "
            "source. Quote or paraphrase these passages and mention the source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."},
                "k": {
                    "type": "integer",
                    "description": "Number of passages to return (default 4).",
                },
            },
            "required": ["query"],
        },
    },
]


class ToolDispatcher:
    """Routes LLM tool calls to a SimulationSession and logs everything."""

    def __init__(self, session: SimulationSession | None = None, session_id: str | None = None):
        self.session = session or SimulationSession()
        self.builder = ProcessBuilder()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_LOG_DIR.mkdir(exist_ok=True)
        self.log_path: Path = RUN_LOG_DIR / f"session_{self.session_id}.jsonl"
        # Knowledge-base passages retrieved during the current turn (for the UI).
        self.last_doc_sources: list[dict] = []

    def _log(self, record: dict[str, Any]) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["biosteam_version"] = biosteam.__version__
        if self.session.is_loaded:
            record["loaded_model"] = self.session.spec.key
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self._dispatch(name, arguments)
            self._log({"tool": name, "arguments": arguments, "result": result})
            return result
        except Exception as exc:  # surfaced back to the LLM as a tool error
            error = {"error": type(exc).__name__, "message": str(exc)}
            self._log({"tool": name, "arguments": arguments, "result": error})
            return error

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        s = self.session
        if name == "list_models":
            return {"models": available_models()}
        if name == "load_model":
            return s.load_model(args["model_key"])
        if name == "list_parameters":
            return {"parameters": s.list_parameters()}
        if name == "set_parameter":
            return s.set_parameter(args["name"], args["value"])
        if name == "reset_parameters":
            return s.reset_parameters()
        if name == "run_simulation":
            return {"metrics": s.run()}
        if name == "run_sensitivity":
            return s.sensitivity(args["parameter"], args["values"])
        if name == "compare_scenarios":
            return s.compare_scenarios(args["scenarios"])
        if name == "run_uncertainty":
            return s.uncertainty(
                args["distributions"], args.get("n_samples", 100)
            )
        if name == "verify_model":
            return s.verify()
        if name == "list_building_blocks":
            return palette()
        if name == "build_process":
            spec = {k: v for k, v in args.items()}
            return self.builder.build(spec)
        if name == "search_docs":
            from .rag import get_retriever

            results = get_retriever().search(args["query"], args.get("k", 4))
            self.last_doc_sources.extend(
                {"title": r["title"], "source": r["source"], "score": r["score"]}
                for r in results
            )
            return {"results": results}
        raise KeyError(f"Unknown tool '{name}'.")
