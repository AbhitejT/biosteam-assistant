"""Declarative registry of curated BioSTEAM models.

Each model is described by:
  - a loader that imports and loads the biorefinery module,
  - a set of vetted, bounded Parameters (each with a getter/setter), and
  - a set of Metrics (each with a getter).

Parameters and metrics operate on the loaded module object, so attribute
paths are resolved lazily at call time (after the module is loaded). This
keeps the registry import cheap and avoids triggering BioSTEAM's expensive
model construction until a model is actually requested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import KG_ETHANOL_PER_GAL


@dataclass
class Parameter:
    name: str
    description: str
    units: str
    getter: Callable[[Any], float]
    setter: Callable[[Any, float], None]
    bounds: tuple[float, float]

    def get(self, module: Any) -> float:
        return float(self.getter(module))

    def set(self, module: Any, value: float) -> None:
        lo, hi = self.bounds
        if not (lo <= value <= hi):
            raise ValueError(
                f"{self.name}={value} is outside the allowed range "
                f"[{lo}, {hi}] ({self.units})."
            )
        self.setter(module, value)


@dataclass
class Metric:
    name: str
    description: str
    units: str
    getter: Callable[[Any], float]

    def get(self, module: Any) -> float:
        return float(self.getter(module))


@dataclass
class ModelSpec:
    key: str
    name: str
    description: str
    loader: Callable[[], Any]
    system_getter: Callable[[Any], Any]
    parameters: dict[str, Parameter] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Corn stover cellulosic ethanol (NREL 2011 design via biorefineries.cornstover)
# --------------------------------------------------------------------------
def _load_cornstover():
    from biorefineries import cornstover as m

    m.load()
    return m


def _cs_set_lifetime(m, years: float) -> None:
    start = m.cornstover_tea.duration[0]
    m.cornstover_tea.duration = (start, start + int(years))


cornstover_spec = ModelSpec(
    key="cornstover",
    name="Corn stover cellulosic ethanol",
    description=(
        "Second-generation (cellulosic) ethanol from corn stover, based on the "
        "NREL biochemical conversion design: dilute-acid pretreatment, enzymatic "
        "saccharification, and co-fermentation of glucose and xylose."
    ),
    loader=_load_cornstover,
    system_getter=lambda m: m.cornstover_sys,
    parameters={
        "feedstock_price": Parameter(
            name="feedstock_price",
            description="Corn stover feedstock purchase price.",
            units="USD/kg",
            getter=lambda m: m.cornstover.price,
            setter=lambda m, v: setattr(m.cornstover, "price", v),
            bounds=(0.0, 0.5),
        ),
        "glucose_to_ethanol_conversion": Parameter(
            name="glucose_to_ethanol_conversion",
            description=(
                "Fraction of glucose converted to ethanol during co-fermentation "
                "(commonly called 'fermentation efficiency')."
            ),
            units="fraction",
            getter=lambda m: m.R303.cofermentation[0].X,
            setter=lambda m, v: setattr(m.R303.cofermentation[0], "X", v),
            bounds=(0.0, 1.0),
        ),
        "xylose_to_ethanol_conversion": Parameter(
            name="xylose_to_ethanol_conversion",
            description="Fraction of xylose converted to ethanol during co-fermentation.",
            units="fraction",
            getter=lambda m: m.R303.cofermentation[4].X,
            setter=lambda m, v: setattr(m.R303.cofermentation[4], "X", v),
            bounds=(0.0, 1.0),
        ),
        "plant_lifetime": Parameter(
            name="plant_lifetime",
            description="Plant operating lifetime used in the discounted cash flow analysis.",
            units="years",
            getter=lambda m: m.cornstover_tea.duration[1] - m.cornstover_tea.duration[0],
            setter=_cs_set_lifetime,
            bounds=(5, 50),
        ),
        "irr": Parameter(
            name="irr",
            description="Internal rate of return targeted when solving for product price.",
            units="fraction",
            getter=lambda m: m.cornstover_tea.IRR,
            setter=lambda m, v: setattr(m.cornstover_tea, "IRR", v),
            bounds=(0.0, 0.5),
        ),
    },
    metrics={
        "mesp_per_gal": Metric(
            name="mesp_per_gal",
            description="Minimum ethanol selling price (the headline TEA metric).",
            units="USD/gal",
            getter=lambda m: m.cornstover_tea.solve_price(m.ethanol) * KG_ETHANOL_PER_GAL,
        ),
        "mesp_per_kg": Metric(
            name="mesp_per_kg",
            description="Minimum ethanol selling price per kilogram.",
            units="USD/kg",
            getter=lambda m: m.cornstover_tea.solve_price(m.ethanol),
        ),
        "ethanol_production": Metric(
            name="ethanol_production",
            description="Ethanol product mass flow rate.",
            units="kg/hr",
            getter=lambda m: m.ethanol.F_mass,
        ),
        "total_capital_investment": Metric(
            name="total_capital_investment",
            description="Total capital investment (TCI).",
            units="USD",
            getter=lambda m: m.cornstover_tea.TCI,
        ),
        "annual_material_cost": Metric(
            name="annual_material_cost",
            description="Annual feedstock and material cost.",
            units="USD/yr",
            getter=lambda m: m.cornstover_tea.material_cost,
        ),
    },
)


# --------------------------------------------------------------------------
# Sugarcane first-generation ethanol
# --------------------------------------------------------------------------
def _load_sugarcane():
    from biorefineries import sugarcane as m

    m.load()
    return m


def _sc_set_lifetime(m, years: float) -> None:
    start = m.sugarcane_tea.duration[0]
    m.sugarcane_tea.duration = (start, start + int(years))


sugarcane_spec = ModelSpec(
    key="sugarcane",
    name="Sugarcane first-generation ethanol",
    description=(
        "First-generation ethanol from sugarcane juice via milling, "
        "fermentation, and distillation, with bagasse-fired cogeneration."
    ),
    loader=_load_sugarcane,
    system_getter=lambda m: m.sugarcane_sys,
    parameters={
        "feedstock_price": Parameter(
            name="feedstock_price",
            description="Sugarcane feedstock purchase price.",
            units="USD/kg",
            getter=lambda m: m.sugarcane.price,
            setter=lambda m, v: setattr(m.sugarcane, "price", v),
            bounds=(0.0, 0.2),
        ),
        "fermentation_efficiency": Parameter(
            name="fermentation_efficiency",
            description="Fraction of glucose converted to ethanol during fermentation.",
            units="fraction",
            getter=lambda m: m.R301.efficiency,
            setter=lambda m, v: setattr(m.R301, "efficiency", v),
            bounds=(0.0, 1.0),
        ),
        "plant_lifetime": Parameter(
            name="plant_lifetime",
            description="Plant operating lifetime used in the discounted cash flow analysis.",
            units="years",
            getter=lambda m: m.sugarcane_tea.duration[1] - m.sugarcane_tea.duration[0],
            setter=_sc_set_lifetime,
            bounds=(5, 50),
        ),
        "irr": Parameter(
            name="irr",
            description="Internal rate of return targeted when solving for product price.",
            units="fraction",
            getter=lambda m: m.sugarcane_tea.IRR,
            setter=lambda m, v: setattr(m.sugarcane_tea, "IRR", v),
            bounds=(0.0, 0.5),
        ),
    },
    metrics={
        "mesp_per_gal": Metric(
            name="mesp_per_gal",
            description="Minimum ethanol selling price (the headline TEA metric).",
            units="USD/gal",
            getter=lambda m: m.sugarcane_tea.solve_price(m.ethanol) * KG_ETHANOL_PER_GAL,
        ),
        "mesp_per_kg": Metric(
            name="mesp_per_kg",
            description="Minimum ethanol selling price per kilogram.",
            units="USD/kg",
            getter=lambda m: m.sugarcane_tea.solve_price(m.ethanol),
        ),
        "ethanol_production": Metric(
            name="ethanol_production",
            description="Ethanol product mass flow rate.",
            units="kg/hr",
            getter=lambda m: m.ethanol.F_mass,
        ),
        "total_capital_investment": Metric(
            name="total_capital_investment",
            description="Total capital investment (TCI).",
            units="USD",
            getter=lambda m: m.sugarcane_tea.TCI,
        ),
    },
)


# --------------------------------------------------------------------------
# Lipidcane: co-production of ethanol and biodiesel from oil-rich cane
# --------------------------------------------------------------------------
def _load_lipidcane():
    from biorefineries import lipidcane as m

    m.load()
    return m


def _lc_set_lifetime(m, years: float) -> None:
    start = m.lipidcane_tea.duration[0]
    m.lipidcane_tea.duration = (start, start + int(years))


lipidcane_spec = ModelSpec(
    key="lipidcane",
    name="Lipidcane ethanol + biodiesel",
    description=(
        "Co-production of ethanol and biodiesel from oil-rich (lipid) sugarcane. "
        "Sugars are fermented to ethanol while the extracted oil is "
        "transesterified to biodiesel; bagasse is burned for cogeneration. The "
        "minimum ethanol selling price is solved with biodiesel sold at its "
        "co-product price."
    ),
    loader=_load_lipidcane,
    system_getter=lambda m: m.lipidcane_sys,
    parameters={
        "feedstock_price": Parameter(
            name="feedstock_price",
            description="Lipidcane feedstock purchase price.",
            units="USD/kg",
            getter=lambda m: m.lipidcane.price,
            setter=lambda m, v: setattr(m.lipidcane, "price", v),
            bounds=(0.0, 0.2),
        ),
        "fermentation_efficiency": Parameter(
            name="fermentation_efficiency",
            description="Fraction of glucose converted to ethanol during fermentation.",
            units="fraction",
            getter=lambda m: m.R301.efficiency,
            setter=lambda m, v: setattr(m.R301, "efficiency", v),
            bounds=(0.0, 1.0),
        ),
        "plant_lifetime": Parameter(
            name="plant_lifetime",
            description="Plant operating lifetime used in the discounted cash flow analysis.",
            units="years",
            getter=lambda m: m.lipidcane_tea.duration[1] - m.lipidcane_tea.duration[0],
            setter=_lc_set_lifetime,
            bounds=(5, 50),
        ),
        "irr": Parameter(
            name="irr",
            description="Internal rate of return targeted when solving for product price.",
            units="fraction",
            getter=lambda m: m.lipidcane_tea.IRR,
            setter=lambda m, v: setattr(m.lipidcane_tea, "IRR", v),
            bounds=(0.0, 0.5),
        ),
    },
    metrics={
        "mesp_per_gal": Metric(
            name="mesp_per_gal",
            description="Minimum ethanol selling price (biodiesel sold at co-product price).",
            units="USD/gal",
            getter=lambda m: m.lipidcane_tea.solve_price(m.ethanol) * KG_ETHANOL_PER_GAL,
        ),
        "mesp_per_kg": Metric(
            name="mesp_per_kg",
            description="Minimum ethanol selling price per kilogram.",
            units="USD/kg",
            getter=lambda m: m.lipidcane_tea.solve_price(m.ethanol),
        ),
        "ethanol_production": Metric(
            name="ethanol_production",
            description="Ethanol product mass flow rate.",
            units="kg/hr",
            getter=lambda m: m.ethanol.F_mass,
        ),
        "biodiesel_production": Metric(
            name="biodiesel_production",
            description="Biodiesel co-product mass flow rate.",
            units="kg/hr",
            getter=lambda m: m.biodiesel.F_mass,
        ),
        "total_capital_investment": Metric(
            name="total_capital_investment",
            description="Total capital investment (TCI).",
            units="USD",
            getter=lambda m: m.lipidcane_tea.TCI,
        ),
    },
)


REGISTRY: dict[str, ModelSpec] = {
    cornstover_spec.key: cornstover_spec,
    sugarcane_spec.key: sugarcane_spec,
    lipidcane_spec.key: lipidcane_spec,
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown model '{key}'. Available models: {sorted(REGISTRY)}."
        )
    return REGISTRY[key]


def list_models() -> list[dict[str, str]]:
    return [
        {"key": s.key, "name": s.name, "description": s.description}
        for s in REGISTRY.values()
    ]
