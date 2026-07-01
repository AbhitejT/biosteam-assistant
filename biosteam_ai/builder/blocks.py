"""The typed building-block palette the assistant may assemble new processes from.

Two allowlists keep from-scratch construction trustworthy:

* ``CHEMICALS`` - chemicals that have complete thermo data and survive a
  vapour-liquid flash, so a built process never crashes on a missing property
  (every entry here was validated against the installed thermosteam).
* ``BLOCK_TYPES`` - the unit-operation templates, each with a declared,
  validated parameter schema. The builder refuses anything outside these
  palettes, which is what makes a generated flowsheet safe to run.
"""
from __future__ import annotations

from typing import Any

# Chemicals validated to load and survive a vapour-liquid flash on the
# installed thermosteam. Keeping the palette curated is the price of never
# handing the user a process that blows up on a thermo-data gap.
CHEMICALS: dict[str, str] = {
    "Water": "Water (H2O).",
    "Ethanol": "Ethanol (C2H6O).",
    "Methanol": "Methanol (CH4O).",
    "Propanol": "1-Propanol (C3H8O).",
    "Butanol": "1-Butanol (C4H10O).",
    "Acetone": "Acetone (C3H6O).",
    "AceticAcid": "Acetic acid (C2H4O2).",
    "EthylAcetate": "Ethyl acetate (C4H8O2).",
    "Acetaldehyde": "Acetaldehyde (C2H4O).",
    "Furfural": "Furfural (C5H4O2).",
    "Glycerol": "Glycerol (C3H8O3).",
    "LacticAcid": "Lactic acid (C3H6O3).",
    "Hexane": "n-Hexane (C6H14).",
    "Octane": "n-Octane (C8H18).",
    "Benzene": "Benzene (C6H6).",
    "Toluene": "Toluene (C7H8).",
    "Ammonia": "Ammonia (NH3).",
    "CO2": "Carbon dioxide (CO2).",
    "CO": "Carbon monoxide (CO).",
    "O2": "Oxygen (O2).",
    "N2": "Nitrogen (N2).",
    "CH4": "Methane (CH4).",
}


# Each block type declares its parameter schema. "required"/"optional" map a
# parameter name to a human-readable spec; the builder validates against these.
BLOCK_TYPES: dict[str, dict[str, Any]] = {
    "mixer": {
        "description": "Combine two or more inlet streams into one outlet.",
        "n_ins": "2+",
        "n_outs": 1,
        "required": {},
        "optional": {},
    },
    "reactor": {
        "description": (
            "Convert reactants to products via one stoichiometric reaction at a "
            "fixed conversion. Mass is conserved exactly. Optionally set the "
            "outlet temperature (K); otherwise the feed temperature is kept."
        ),
        "n_ins": 1,
        "n_outs": 1,
        "required": {
            "reaction": "Reaction string, e.g. 'Ethanol + AceticAcid -> EthylAcetate + Water'.",
            "reactant": "Name of the limiting reactant the conversion applies to.",
            "conversion": "Fractional conversion of the reactant in [0, 1].",
        },
        "optional": {
            "T": "Outlet temperature in K.",
            "P": "Outlet pressure in Pa.",
        },
    },
    "flash": {
        "description": (
            "Flash one inlet into a vapour and a liquid outlet by vapour-liquid "
            "equilibrium. Provide the vapour fraction V in [0, 1]."
        ),
        "n_ins": 1,
        "n_outs": 2,
        "required": {
            "V": "Molar vapour fraction in [0, 1] (outs order: vapour, liquid).",
        },
        "optional": {
            "P": "Operating pressure in Pa (default 101325).",
        },
    },
    "splitter": {
        "description": (
            "Split one inlet into two outlets of identical composition by a "
            "fixed split fraction (outs order: top split, remainder)."
        ),
        "n_ins": 1,
        "n_outs": 2,
        "required": {
            "split": "Fraction of each component sent to the first outlet, in [0, 1].",
        },
        "optional": {},
    },
    "heater": {
        "description": (
            "Heat or cool one inlet to a target temperature (a HXutility). "
            "Composition is unchanged; duty and cost are computed."
        ),
        "n_ins": 1,
        "n_outs": 1,
        "required": {
            "T": "Target outlet temperature in K.",
        },
        "optional": {
            "P": "Outlet pressure in Pa.",
        },
    },
}


def palette() -> dict[str, Any]:
    """Machine-readable description of everything that can be assembled."""
    return {
        "chemicals": CHEMICALS,
        "block_types": BLOCK_TYPES,
        "notes": (
            "Streams are referenced by name. A unit's 'ins' must already exist "
            "(a feed, or an earlier unit's 'outs'); its 'outs' create new "
            "streams. Flows are in kmol/hr, temperatures in K, pressures in Pa."
        ),
    }
