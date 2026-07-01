"""Shared correctness checks for BioSTEAM systems.

These functions operate on a plain BioSTEAM ``System`` object and return
structured check dicts of the form::

    {"name", "severity", "status", "detail"}

where ``severity`` is "error" or "warning" and ``status`` is
"pass" | "warn" | "fail". Both the curated registry models
(:mod:`biosteam_ai.engine`) and the from-scratch builder
(:mod:`biosteam_ai.builder`) compose these into a single report via
:func:`summarize`, so "verified" means the same thing everywhere.
"""
from __future__ import annotations

import contextlib
import io
import warnings
from typing import Any

import numpy as np


@contextlib.contextmanager
def quiet():
    """Suppress BioSTEAM's verbose cost/convergence warnings and stdout."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            yield


def reaction_imbalances(obj: Any) -> list[float]:
    """Mass imbalance |sum(stoichiometry * MW)| (g/mol) for each reaction in a
    reaction object (single, parallel, or series). Empty if obj is not one."""
    try:
        chemicals = obj.chemicals
        stoich = obj.stoichiometry
    except AttributeError:
        return []
    try:
        mw = np.asarray(chemicals.MW)
        arr = stoich.to_array() if hasattr(stoich, "to_array") else np.asarray(stoich)
    except Exception:
        return []
    if arr.ndim == 1:
        return [abs(float((arr * mw).sum()))]
    return [abs(float((row * mw).sum())) for row in arr]


def check_mass_balance(sys: Any, mass_tol: float = 0.02) -> dict[str, Any]:
    """Per-unit mass balance closure: total mass in vs. out for each unit."""
    worst, worst_u = 0.0, None
    for u in sys.units:
        fin = sum(s.F_mass for s in u.ins)
        fout = sum(s.F_mass for s in u.outs)
        if fin > 1e-9:
            err = abs(fin - fout) / fin
            if err > worst:
                worst, worst_u = err, u.ID
    return {
        "name": "Per-unit mass balance",
        "severity": "warning",
        "status": "pass" if worst <= mass_tol else "warn",
        "detail": (
            f"Worst per-unit imbalance {worst * 100:.3f}% "
            f"at {worst_u} (tolerance {mass_tol * 100:.1f}%)."
            + ("" if worst <= mass_tol
               else " Flag for review: a unit does not conserve mass "
                    "(often a recycle-tear artifact).")
        ),
    }


def check_negative_flows(sys: Any) -> dict[str, Any]:
    """No stream may carry a negative component flow."""
    neg = [s.ID for s in sys.streams if (s.mol.to_array() < -1e-9).any()]
    return {
        "name": "No negative flows",
        "severity": "error",
        "status": "pass" if not neg else "fail",
        "detail": (
            "All stream component flows are non-negative."
            if not neg else
            f"{len(neg)} stream(s) have negative flows: {neg[:5]}"
        ),
    }


def check_reaction_balance(sys: Any, reaction_tol: float = 0.5) -> dict[str, Any]:
    """Reaction stoichiometry should conserve mass (diagnostic warning)."""
    imbalances: list[float] = []
    for u in sys.units:
        for obj in vars(u).values():
            imbalances.extend(reaction_imbalances(obj))
    n_rxn = len(imbalances)
    off = [d for d in imbalances if d > reaction_tol]
    worst_rxn = max(imbalances) if imbalances else 0.0
    return {
        "name": "Reaction mass balance",
        "severity": "warning",
        "status": "pass" if not off else "warn",
        "detail": (
            f"{n_rxn - len(off)}/{n_rxn} reactions conserve mass; "
            f"{len(off)} off by up to {worst_rxn:.2f} g/mol."
            if n_rxn else "No reactions found."
        ),
    }


def summarize(checks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Roll a list of checks into an overall pass/warn/fail report."""
    has_fail = any(c["status"] == "fail" for c in checks)
    has_warn = any(c["status"] == "warn" for c in checks)
    overall = "fail" if has_fail else ("warn" if has_warn else "pass")
    return {"overall": overall, "checks": checks, **extra}
