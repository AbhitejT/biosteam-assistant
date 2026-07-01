"""Simulation session: a safe, stateful wrapper around one BioSTEAM model.

This is the controlled execution layer. It exposes only vetted operations
(load, inspect, set bounded parameters, simulate, read metrics, sensitivity)
and records a provenance log of every state change and run.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .models import ModelSpec, get_model_spec, list_models
from .verification import (
    check_mass_balance,
    check_negative_flows,
    check_reaction_balance,
    quiet as _quiet,
    summarize,
)


class SimulationSession:
    def __init__(self) -> None:
        self.spec: ModelSpec | None = None
        self._module: Any = None
        self._baseline: dict[str, float] = {}
        # Most recent chartable result (for the UI to render). Shape:
        # {"kind": "sensitivity"|"comparison"|"uncertainty", "data": ...}
        self.last_artifact: dict[str, Any] | None = None
        # Most recent verification report (for the UI to render).
        self.last_verification: dict[str, Any] | None = None

    # -- model loading -----------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self.spec is not None

    def _require_model(self) -> ModelSpec:
        if self.spec is None:
            raise RuntimeError("No model is loaded. Call load_model first.")
        return self.spec

    def load_model(self, key: str) -> dict[str, Any]:
        spec = get_model_spec(key)
        with _quiet():
            self._module = spec.loader()
        self.spec = spec
        self.last_artifact = None
        self.last_verification = None
        # Snapshot baseline values of all registered parameters for reset().
        self._baseline = {
            name: p.get(self._module) for name, p in spec.parameters.items()
        }
        return {
            "model": spec.key,
            "name": spec.name,
            "description": spec.description,
            "parameters": self.list_parameters(),
        }

    # -- parameters --------------------------------------------------------
    def list_parameters(self) -> list[dict[str, Any]]:
        spec = self._require_model()
        out = []
        for name, p in spec.parameters.items():
            out.append(
                {
                    "name": name,
                    "description": p.description,
                    "units": p.units,
                    "bounds": list(p.bounds),
                    "current_value": p.get(self._module),
                }
            )
        return out

    def get_parameter(self, name: str) -> float:
        spec = self._require_model()
        if name not in spec.parameters:
            raise KeyError(f"Unknown parameter '{name}'.")
        return spec.parameters[name].get(self._module)

    def set_parameter(self, name: str, value: float) -> dict[str, Any]:
        spec = self._require_model()
        if name not in spec.parameters:
            raise KeyError(
                f"Unknown parameter '{name}'. "
                f"Allowed: {sorted(spec.parameters)}."
            )
        previous = spec.parameters[name].get(self._module)
        spec.parameters[name].set(self._module, value)
        return {"parameter": name, "previous_value": previous, "new_value": value}

    def reset_parameters(self) -> dict[str, Any]:
        spec = self._require_model()
        for name, value in self._baseline.items():
            spec.parameters[name].set(self._module, value)
        return {"status": "reset_to_baseline", "parameters": self.list_parameters()}

    # -- simulation & metrics ---------------------------------------------
    def run(self) -> dict[str, Any]:
        spec = self._require_model()
        with _quiet():
            spec.system_getter(self._module).simulate()
        return self.get_metrics()

    def get_metrics(self) -> dict[str, Any]:
        spec = self._require_model()
        results = {}
        with _quiet():
            for name, metric in spec.metrics.items():
                results[name] = {
                    "value": metric.get(self._module),
                    "units": metric.units,
                    "description": metric.description,
                }
        return results

    def _flat_metrics(self) -> dict[str, float]:
        return {k: m["value"] for k, m in self.run().items()}

    def sensitivity(self, parameter: str, values: list[float]) -> dict[str, Any]:
        """Sweep one parameter across values, re-simulating at each point."""
        spec = self._require_model()
        if parameter not in spec.parameters:
            raise KeyError(f"Unknown parameter '{parameter}'.")
        original = spec.parameters[parameter].get(self._module)
        rows = []
        try:
            for v in values:
                self.set_parameter(parameter, v)
                rows.append(
                    {"parameter_value": v, "metrics": self._flat_metrics()}
                )
        finally:
            self.set_parameter(parameter, original)
            self.run()
        units = {k: m["units"] for k, m in self.get_metrics().items()}
        result = {"parameter": parameter, "units": spec.parameters[parameter].units, "sweep": rows}
        self.last_artifact = {
            "kind": "sensitivity",
            "data": {**result, "metric_units": units},
        }
        return result

    def compare_scenarios(
        self, scenarios: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Run named scenarios (each a dict of parameter overrides) from
        baseline and return metrics plus deltas vs. baseline."""
        self._require_model()
        self.reset_parameters()
        baseline = self._flat_metrics()
        out: dict[str, Any] = {}
        try:
            for name, params in scenarios.items():
                self.reset_parameters()
                applied = {}
                for k, v in params.items():
                    self.set_parameter(k, v)
                    applied[k] = v
                metrics = self._flat_metrics()
                out[name] = {
                    "parameters": applied,
                    "metrics": metrics,
                    "delta_vs_baseline": {
                        k: metrics[k] - baseline[k] for k in metrics
                    },
                    "pct_change_vs_baseline": {
                        k: (100.0 * (metrics[k] - baseline[k]) / baseline[k]
                            if baseline[k] else None)
                        for k in metrics
                    },
                }
        finally:
            self.reset_parameters()
            self.run()
        units = {k: m["units"] for k, m in self.get_metrics().items()}
        result = {"baseline": baseline, "scenarios": out}
        self.last_artifact = {
            "kind": "comparison",
            "data": {**result, "metric_units": units},
        }
        return result

    def uncertainty(
        self,
        distributions: list[dict[str, Any]],
        n_samples: int = 100,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Monte Carlo over one or more parameters.

        Each distribution is {"name", "low", "high", optional "kind"
        ('uniform' default or 'triangular'), optional "mode"}. Returns
        summary statistics (mean/std/percentiles) per metric. Full samples
        are stored on last_artifact for charting but not returned here.
        """
        spec = self._require_model()
        if not distributions:
            raise ValueError("Provide at least one parameter distribution.")
        n_samples = int(max(1, min(n_samples, 1000)))
        for d in distributions:
            name = d["name"]
            if name not in spec.parameters:
                raise KeyError(f"Unknown parameter '{name}'.")
            lo, hi = spec.parameters[name].bounds
            if d["low"] < lo or d["high"] > hi:
                raise ValueError(
                    f"{name} range [{d['low']}, {d['high']}] exceeds allowed "
                    f"bounds [{lo}, {hi}]."
                )
            if d["low"] > d["high"]:
                raise ValueError(f"{name}: low must be <= high.")

        rng = np.random.default_rng(seed)
        metric_names = list(spec.metrics)
        samples: dict[str, list[float]] = {m: [] for m in metric_names}
        param_draws: dict[str, list[float]] = {d["name"]: [] for d in distributions}

        self.reset_parameters()
        try:
            for _ in range(n_samples):
                for d in distributions:
                    if d.get("kind") == "triangular":
                        mode = d.get("mode", (d["low"] + d["high"]) / 2)
                        v = float(rng.triangular(d["low"], mode, d["high"]))
                    else:
                        v = float(rng.uniform(d["low"], d["high"]))
                    self.set_parameter(d["name"], v)
                    param_draws[d["name"]].append(v)
                flat = self._flat_metrics()
                for m in metric_names:
                    samples[m].append(flat[m])
        finally:
            self.reset_parameters()
            self.run()

        summary = {}
        for m, vals in samples.items():
            arr = np.asarray(vals, dtype=float)
            summary[m] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "p5": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        units = {k: v["units"] for k, v in self.get_metrics().items()}
        self.last_artifact = {
            "kind": "uncertainty",
            "data": {
                "n_samples": n_samples,
                "distributions": distributions,
                "samples": samples,
                "param_draws": param_draws,
                "metric_units": units,
            },
        }
        return {
            "n_samples": n_samples,
            "parameters": [d["name"] for d in distributions],
            "summary": summary,
            "metric_units": units,
        }


    def verify(
        self, mass_tol: float = 0.02, reaction_tol: float = 0.5
    ) -> dict[str, Any]:
        """Check that the simulated model is physically and economically sane,
        not merely that it ran. Returns a structured validation report.

        Checks (severity):
          - Per-unit mass balance closure (warning): inputs vs outputs per unit.
          - No negative component flows (error).
          - Reaction mass balance (warning): reaction stoichiometry conserves mass.
          - Output plausibility (error): metrics finite, MESP/production sensible.
        """
        spec = self._require_model()
        sys = spec.system_getter(self._module)

        with _quiet():
            checks = [
                check_mass_balance(sys, mass_tol),
                check_negative_flows(sys),
                check_reaction_balance(sys, reaction_tol),
                self._output_plausibility_check(),
            ]

        report = summarize(checks, model=spec.key)
        self.last_verification = report
        return report

    def _output_plausibility_check(self) -> dict[str, Any]:
        metrics = self.get_metrics()
        issues = []
        for name, m in metrics.items():
            v = m["value"]
            if v != v or v in (float("inf"), float("-inf")):
                issues.append(f"{name} is not finite")
        mesp = metrics.get("mesp_per_gal", {}).get("value")
        if mesp is not None and not (0 < mesp < 50):
            issues.append(f"MESP {mesp:.2f} USD/gal outside plausible range")
        for prod in ("ethanol_production", "biodiesel_production"):
            if prod in metrics and metrics[prod]["value"] <= 0:
                issues.append(f"{prod} is not positive")
        return {
            "name": "Output plausibility",
            "severity": "error",
            "status": "pass" if not issues else "fail",
            "detail": "All metrics finite and within plausible ranges."
                      if not issues else "; ".join(issues),
        }


def available_models() -> list[dict[str, str]]:
    return list_models()
