"""Assemble, simulate, and verify a small process from typed building blocks.

This is the guarded model-*builder*: given a declarative spec (chemicals, feed
streams, and a list of unit blocks wired by stream name), it constructs a real
BioSTEAM flowsheet, simulates it, and runs the same verification layer used for
the curated registry models. Everything is validated against the allowlists in
:mod:`biosteam_ai.builder.blocks` first, so the assistant can only build
processes that are known to run.
"""
from __future__ import annotations

import itertools
import re
from typing import Any

import biosteam as bst
import thermosteam as tmo

from ..verification import (
    check_mass_balance,
    check_negative_flows,
    check_reaction_balance,
    quiet,
    summarize,
)
from .blocks import BLOCK_TYPES, CHEMICALS

_flowsheet_counter = itertools.count(1)


class BuilderError(ValueError):
    """Raised for an invalid process spec, with a user-facing message."""


class SimpleTEA(bst.TEA):
    """A minimal but real techno-economic model for a built process.

    Capital is estimated from installed equipment cost via a Lang factor, and
    fixed operating cost is taken as a fraction of fixed capital investment.
    This lets a from-scratch flowsheet return an actual minimum product selling
    price (via :meth:`solve_price`) using standard financial assumptions, rather
    than only equipment cost. It is intentionally simpler than the bespoke TEA
    classes in the curated registry models -- those remain the reference for
    fully-validated selling prices.
    """

    def __init__(
        self,
        system,
        IRR: float = 0.10,
        plant_years: int = 20,
        income_tax: float = 0.21,
        operating_days: float = 330,
        lang_factor: float = 3.0,
        FOC_over_FCI: float = 0.05,
    ):
        start = 2023
        super().__init__(
            system,
            IRR=IRR,
            duration=(start, start + int(plant_years)),
            depreciation="MACRS7",
            income_tax=income_tax,
            operating_days=operating_days,
            lang_factor=lang_factor,
            construction_schedule=(0.4, 0.6),
            startup_months=0,
            startup_FOCfrac=1,
            startup_VOCfrac=1,
            startup_salesfrac=1,
            WC_over_FCI=0.05,
            finance_interest=0,
            finance_years=0,
            finance_fraction=0,
        )
        self.FOC_over_FCI = FOC_over_FCI

    def _FOC(self, FCI: float) -> float:
        return FCI * self.FOC_over_FCI


class ConversionReactor(bst.Unit):
    """Minimal, predictable stoichiometric reactor block.

    Applies one reaction at a fixed conversion and (optionally) sets the outlet
    temperature/pressure. Mass is conserved exactly and there is no brittle
    sizing routine, so it simulates reliably for any feed -- unlike BioSTEAM's
    stirred-tank reactor classes, whose costing assumes a liquid recirculation
    loop and crashes on gas-phase or isothermal duties.
    """

    _N_ins = 1
    _N_outs = 1

    def _init(self, reaction, T=None, P=None):  # noqa: D401 - BioSTEAM hook
        self.reaction = reaction
        self.T = T
        self.P = P

    def _run(self):
        feed = self.ins[0]
        out = self.outs[0]
        out.copy_like(feed)
        self.reaction(out)
        if self.T is not None:
            out.T = self.T
        if self.P is not None:
            out.P = self.P


class ProcessBuilder:
    """Stateful holder for the most-recently-built process.

    The whole flowsheet is described in one declarative spec and built
    atomically by :meth:`build`; the resulting results/verification are kept for
    the UI to render.
    """

    def __init__(self) -> None:
        self.last_spec: dict[str, Any] | None = None
        self.last_results: dict[str, Any] | None = None
        self.last_verification: dict[str, Any] | None = None
        self._sys: Any = None

    # -- public API --------------------------------------------------------
    def build(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Validate, construct, simulate, and verify a process in one shot.

        Returns ``{"spec", "results", "verification"}``. Raises
        :class:`BuilderError` with an explanatory message on invalid input.
        """
        chemicals, feeds, units = self._validate_spec(spec)
        name = str(spec.get("name") or "custom process")
        product = spec.get("product")
        economics = self._validate_economics(spec.get("economics"))

        with quiet():
            sys, streams = self._construct(chemicals, feeds, units)
            sys.simulate()
            econ = None
            if product is not None:
                econ = self._economics(sys, streams[product], product, economics)

        self._sys = sys
        self.last_spec = spec
        self.last_results = self._results(name, sys, streams, feeds, units, econ)
        self.last_verification = self._verify(name, sys, econ)
        return {
            "spec": {"name": name, "chemicals": chemicals,
                     "n_feeds": len(feeds), "n_units": len(units)},
            "results": self.last_results,
            "verification": self.last_verification,
        }

    @staticmethod
    def _validate_economics(economics: dict[str, Any] | None) -> dict[str, float]:
        """Merge user overrides onto default financial assumptions."""
        defaults = {
            "IRR": 0.10,
            "plant_years": 20,
            "income_tax": 0.21,
            "operating_days": 330,
            "lang_factor": 3.0,
            "FOC_over_FCI": 0.05,
        }
        if not economics:
            return defaults
        bounds = {
            "IRR": (0.0, 1.0),
            "plant_years": (1, 100),
            "income_tax": (0.0, 1.0),
            "operating_days": (1, 365),
            "lang_factor": (1.0, 10.0),
            "FOC_over_FCI": (0.0, 1.0),
        }
        for key, val in economics.items():
            if key not in defaults:
                raise BuilderError(
                    f"Unknown economics key '{key}'. Allowed: {sorted(defaults)}."
                )
            lo, hi = bounds[key]
            if not (lo <= float(val) <= hi):
                raise BuilderError(
                    f"economics '{key}'={val} outside allowed range [{lo}, {hi}]."
                )
            defaults[key] = float(val)
        return defaults

    def _economics(
        self, sys: Any, product_stream: Any, product_name: str,
        economics: dict[str, float],
    ) -> dict[str, Any]:
        """Build a SimpleTEA and solve the minimum product selling price."""
        tea = SimpleTEA(sys, **economics)
        msp = float(tea.solve_price(product_stream))
        return {
            "product": product_name,
            "assumptions": economics,
            "min_selling_price_usd_per_kg": round(msp, 4),
            "total_capital_investment_usd": round(float(tea.TCI), 2),
            "fixed_operating_cost_usd_per_yr": round(float(tea.FOC), 2),
            "material_cost_usd_per_yr": round(float(tea.material_cost), 2),
            "product_flow_kg_per_hr": round(float(product_stream.F_mass), 3),
        }

    # -- validation --------------------------------------------------------
    def _validate_spec(
        self, spec: dict[str, Any]
    ) -> tuple[list[str], list[dict], list[dict]]:
        if not isinstance(spec, dict):
            raise BuilderError("Process spec must be an object.")

        chemicals = spec.get("chemicals")
        if not chemicals or not isinstance(chemicals, list):
            raise BuilderError("Provide a non-empty 'chemicals' list.")
        bad = [c for c in chemicals if c not in CHEMICALS]
        if bad:
            raise BuilderError(
                f"Chemical(s) not in the allowlist: {bad}. "
                f"Allowed: {sorted(CHEMICALS)}."
            )
        chem_set = set(chemicals)

        feeds = spec.get("feeds") or []
        if not feeds:
            raise BuilderError("Provide at least one feed stream.")
        stream_names: set[str] = set()
        for f in feeds:
            nm = f.get("name")
            if not nm:
                raise BuilderError("Every feed needs a 'name'.")
            if nm in stream_names:
                raise BuilderError(f"Duplicate stream name '{nm}'.")
            flows = f.get("flows") or {}
            if not flows:
                raise BuilderError(f"Feed '{nm}' has no 'flows'.")
            for c in flows:
                if c not in chem_set:
                    raise BuilderError(
                        f"Feed '{nm}' uses '{c}', which is not in this "
                        f"process's chemicals."
                    )
            if f.get("price") is not None and float(f["price"]) < 0:
                raise BuilderError(f"Feed '{nm}': price must be >= 0.")
            stream_names.add(nm)

        units = spec.get("units") or []
        if not units:
            raise BuilderError("Provide at least one unit block.")
        consumed: set[str] = set()
        produced: set[str] = set()
        unit_ids: set[str] = set()
        for u in units:
            uid = u.get("id")
            utype = u.get("type")
            if not uid:
                raise BuilderError("Every unit needs an 'id'.")
            if uid in unit_ids:
                raise BuilderError(f"Duplicate unit id '{uid}'.")
            unit_ids.add(uid)
            if utype not in BLOCK_TYPES:
                raise BuilderError(
                    f"Unit '{uid}' has unknown type '{utype}'. "
                    f"Allowed: {sorted(BLOCK_TYPES)}."
                )
            ins = u.get("ins") or []
            outs = u.get("outs") or []
            self._check_arity(uid, utype, ins, outs)
            for s in ins:
                if s not in stream_names:
                    raise BuilderError(
                        f"Unit '{uid}' reads stream '{s}' before it is "
                        f"produced. Order units so inputs exist first; "
                        f"recycles are not yet supported."
                    )
                consumed.add(s)
            for s in outs:
                if s in stream_names:
                    raise BuilderError(
                        f"Unit '{uid}' output '{s}' reuses an existing stream "
                        f"name; output names must be new."
                    )
                stream_names.add(s)
                produced.add(s)
            self._validate_unit_params(uid, utype, u, chem_set)

        product = spec.get("product")
        if product is not None:
            terminal = produced - consumed
            if product not in terminal:
                raise BuilderError(
                    f"'product' must be a terminal product stream "
                    f"(one of {sorted(terminal)}), got '{product}'."
                )

        return list(chemicals), feeds, units

    @staticmethod
    def _check_arity(uid: str, utype: str, ins: list, outs: list) -> None:
        spec = BLOCK_TYPES[utype]
        n_in, n_out = spec["n_ins"], spec["n_outs"]
        if n_in == "2+":
            if len(ins) < 2:
                raise BuilderError(f"Unit '{uid}' ({utype}) needs >= 2 inputs.")
        elif len(ins) != n_in:
            raise BuilderError(
                f"Unit '{uid}' ({utype}) needs exactly {n_in} input(s), got {len(ins)}."
            )
        if len(outs) != n_out:
            raise BuilderError(
                f"Unit '{uid}' ({utype}) needs exactly {n_out} output(s), got {len(outs)}."
            )

    @staticmethod
    def _validate_unit_params(
        uid: str, utype: str, u: dict, chem_set: set[str]
    ) -> None:
        for key in BLOCK_TYPES[utype]["required"]:
            if u.get(key) is None:
                raise BuilderError(f"Unit '{uid}' ({utype}) is missing '{key}'.")
        if utype == "reactor":
            if not (0.0 <= float(u["conversion"]) <= 1.0):
                raise BuilderError(f"Unit '{uid}': conversion must be in [0, 1].")
            if u["reactant"] not in chem_set:
                raise BuilderError(
                    f"Unit '{uid}': reactant '{u['reactant']}' is not in this "
                    f"process's chemicals."
                )
            species = set(re.findall(r"[A-Za-z][A-Za-z0-9]*", str(u["reaction"])))
            unknown = species - chem_set
            if unknown:
                raise BuilderError(
                    f"Unit '{uid}': reaction references {sorted(unknown)} not in "
                    f"this process's chemicals."
                )
        if utype == "flash" and not (0.0 <= float(u["V"]) <= 1.0):
            raise BuilderError(f"Unit '{uid}': V must be in [0, 1].")
        if utype == "splitter" and not (0.0 <= float(u["split"]) <= 1.0):
            raise BuilderError(f"Unit '{uid}': split must be in [0, 1].")

    # -- construction ------------------------------------------------------
    def _construct(
        self, chemicals: list[str], feeds: list[dict], units: list[dict]
    ) -> tuple[Any, dict[str, Any]]:
        flowsheet = bst.Flowsheet(f"builder_{next(_flowsheet_counter)}")
        bst.main_flowsheet.set_flowsheet(flowsheet)
        bst.settings.set_thermo(chemicals)

        streams: dict[str, Any] = {}
        for f in feeds:
            stream = bst.Stream(
                f["name"],
                T=float(f.get("T", 298.15)),
                P=float(f.get("P", 101325.0)),
                units="kmol/hr",
                **{k: float(v) for k, v in f["flows"].items()},
            )
            if f.get("price") is not None:
                stream.price = float(f["price"])  # USD/kg, feeds the TEA
            streams[f["name"]] = stream

        unit_objs = []
        for u in units:
            ins = [streams[s] for s in u["ins"]]
            out_names = u["outs"]
            unit_objs.append(self._make_unit(u, ins, out_names, streams))

        sys = bst.System.from_units(f"{flowsheet.ID}_sys", unit_objs)
        return sys, streams

    def _make_unit(
        self, u: dict, ins: list, out_names: list[str], streams: dict
    ) -> Any:
        utype, uid = u["type"], u["id"]

        def register(outs):
            for name, stream in zip(out_names, outs):
                stream.ID = name
                streams[name] = stream
            return outs

        if utype == "mixer":
            unit = bst.Mixer(uid, ins=ins, outs=out_names[0])
        elif utype == "reactor":
            rxn = tmo.Reaction(
                u["reaction"], reactant=u["reactant"], X=float(u["conversion"])
            )
            kwargs = {}
            if u.get("T") is not None:
                kwargs["T"] = float(u["T"])
            if u.get("P") is not None:
                kwargs["P"] = float(u["P"])
            unit = ConversionReactor(uid, ins=ins[0], outs=out_names[0],
                                     reaction=rxn, **kwargs)
        elif utype == "flash":
            unit = bst.Flash(
                uid, ins=ins[0], outs=tuple(out_names),
                V=float(u["V"]), P=float(u.get("P", 101325.0)),
            )
        elif utype == "splitter":
            unit = bst.Splitter(
                uid, ins=ins[0], outs=tuple(out_names), split=float(u["split"])
            )
        elif utype == "heater":
            kwargs = {"T": float(u["T"])}
            if u.get("P") is not None:
                kwargs["P"] = float(u["P"])
            unit = bst.HXutility(uid, ins=ins[0], outs=out_names[0], **kwargs)
        else:  # pragma: no cover - guarded by validation
            raise BuilderError(f"Unsupported unit type '{utype}'.")

        register(unit.outs)
        return unit

    # -- results & verification -------------------------------------------
    def _results(
        self, name: str, sys: Any, streams: dict, feeds: list, units: list,
        econ: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        feed_names = {f["name"] for f in feeds}
        consumed = {s for u in units for s in u["ins"]}
        produced = {s for u in units for s in u["outs"]}
        product_names = produced - consumed

        def stream_row(nm: str) -> dict[str, Any]:
            s = streams[nm]
            comps = {
                c: round(float(s.imol[c]), 4)
                for c in s.chemicals.IDs
                if float(s.imol[c]) > 1e-9
            }
            return {
                "name": nm,
                "phase": s.phase,
                "T_K": round(float(s.T), 2),
                "P_Pa": round(float(s.P), 1),
                "F_mass_kg_hr": round(float(s.F_mass), 3),
                "F_mol_kmol_hr": round(float(s.F_mol), 4),
                "flows_kmol_hr": comps,
            }

        unit_rows = []
        total_cost = 0.0
        for u, obj in zip(units, sys.units):
            try:
                cost = float(obj.installed_cost)
            except Exception:
                cost = None
            if cost:
                total_cost += cost
            unit_rows.append({
                "id": u["id"], "type": u["type"],
                "installed_cost_usd": round(cost, 2) if cost is not None else None,
            })

        results = {
            "name": name,
            "feeds": [stream_row(n) for n in sorted(feed_names)],
            "products": [stream_row(n) for n in sorted(product_names)],
            "intermediates": [
                stream_row(n) for n in sorted(produced & consumed)
            ],
            "units": unit_rows,
            "total_installed_equipment_cost_usd": round(total_cost, 2),
        }
        if econ is not None:
            results["economics"] = econ
        return results

    def _verify(
        self, name: str, sys: Any, econ: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with quiet():
            checks = [
                check_mass_balance(sys),
                check_negative_flows(sys),
                check_reaction_balance(sys),
                self._plausibility_check(sys),
            ]
            if econ is not None:
                checks.append(self._economic_check(econ))
        return summarize(checks, model=name)

    @staticmethod
    def _economic_check(econ: dict[str, Any]) -> dict[str, Any]:
        msp = econ["min_selling_price_usd_per_kg"]
        issues = []
        if msp != msp or msp in (float("inf"), float("-inf")):
            issues.append("minimum selling price is not finite")
        elif msp <= 0:
            issues.append(f"minimum selling price {msp} is not positive")
        elif msp > 1e5:
            issues.append(f"minimum selling price {msp} USD/kg implausibly large")
        return {
            "name": "Economic plausibility",
            "severity": "error",
            "status": "pass" if not issues else "fail",
            "detail": (
                f"Minimum selling price {msp} USD/kg for '{econ['product']}' "
                f"is finite and positive."
                if not issues else "; ".join(issues)
            ),
        }

    @staticmethod
    def _plausibility_check(sys: Any) -> dict[str, Any]:
        issues = []
        for s in sys.streams:
            f = float(s.F_mass)
            if f != f or f in (float("inf"), float("-inf")):
                issues.append(f"stream {s.ID} flow is not finite")
        terminal = [s for s in sys.streams if not s.sink]
        if terminal and all(float(s.F_mass) <= 1e-9 for s in terminal):
            issues.append("no product stream carries positive mass flow")
        return {
            "name": "Output plausibility",
            "severity": "error",
            "status": "pass" if not issues else "fail",
            "detail": "All stream flows finite with non-zero products."
                      if not issues else "; ".join(issues),
        }
