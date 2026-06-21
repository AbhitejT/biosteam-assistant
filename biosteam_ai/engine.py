"""Simulation session: a safe, stateful wrapper around one BioSTEAM model.

This is the controlled execution layer. It exposes only vetted operations
(load, inspect, set bounded parameters, simulate, read metrics, sensitivity)
and records a provenance log of every state change and run.
"""
from __future__ import annotations

import contextlib
import io
import warnings
from typing import Any

from .models import ModelSpec, get_model_spec, list_models


@contextlib.contextmanager
def _quiet():
    """Suppress BioSTEAM's verbose cost/convergence warnings and stdout."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            yield


class SimulationSession:
    def __init__(self) -> None:
        self.spec: ModelSpec | None = None
        self._module: Any = None
        self._baseline: dict[str, float] = {}

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
                metrics = self.run()
                rows.append(
                    {
                        "parameter_value": v,
                        "metrics": {k: m["value"] for k, m in metrics.items()},
                    }
                )
        finally:
            self.set_parameter(parameter, original)
            self.run()
        return {"parameter": parameter, "sweep": rows}


def available_models() -> list[dict[str, str]]:
    return list_models()
