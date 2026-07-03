"""Tests for the from-scratch process builder.

These assert that a process assembled from typed blocks actually simulates,
conserves mass, costs equipment, and that invalid specs are rejected with
clear errors. No Anthropic API key is required.
"""
import pytest

from biosteam_ai.builder import BLOCK_TYPES, CHEMICALS, ProcessBuilder, palette
from biosteam_ai.builder.process_builder import BuilderError


@pytest.fixture
def builder():
    return ProcessBuilder()


ESTERIFICATION = {
    "name": "ethyl acetate esterification",
    "chemicals": ["Ethanol", "AceticAcid", "EthylAcetate", "Water"],
    "feeds": [{"name": "feed", "flows": {"Ethanol": 100, "AceticAcid": 100}, "T": 350}],
    "units": [
        {
            "id": "R1", "type": "reactor", "ins": ["feed"], "outs": ["crude"],
            "reaction": "Ethanol + AceticAcid -> EthylAcetate + Water",
            "reactant": "Ethanol", "conversion": 0.6, "T": 350,
        },
        {"id": "F1", "type": "flash", "ins": ["crude"], "outs": ["vapor", "liquid"], "V": 0.5},
    ],
}


def test_palette_reports_chemicals_and_blocks():
    p = palette()
    assert set(p["chemicals"]) == set(CHEMICALS)
    assert set(p["block_types"]) == set(BLOCK_TYPES)
    assert {"mixer", "reactor", "flash", "splitter", "heater"} <= set(BLOCK_TYPES)


def test_build_simulates_and_verifies(builder):
    out = builder.build(ESTERIFICATION)
    assert out["verification"]["overall"] == "pass"
    assert out["results"]["total_installed_equipment_cost_usd"] > 0
    assert {p["name"] for p in out["results"]["products"]} == {"vapor", "liquid"}


def test_reactor_conserves_mass_and_hits_conversion(builder):
    builder.build(ESTERIFICATION)
    products = builder.last_results["products"]
    # 0.6 conversion of 100 kmol/hr Ethanol -> 60 kmol/hr EthylAcetate total.
    ea = sum(p["flows_kmol_hr"].get("EthylAcetate", 0.0) for p in products)
    assert ea == pytest.approx(60.0, abs=0.5)
    # System mass in == mass out.
    feed_mass = sum(f["F_mass_kg_hr"] for f in builder.last_results["feeds"])
    prod_mass = sum(p["F_mass_kg_hr"] for p in products)
    assert feed_mass == pytest.approx(prod_mass, rel=1e-4)


def test_verification_flags_unbalanced_reaction(builder):
    # Ethanol -> Acetaldehyde + Water does NOT conserve mass; expect a warning.
    spec = {
        "name": "bad reaction",
        "chemicals": ["Ethanol", "Acetaldehyde", "Water"],
        "feeds": [{"name": "f", "flows": {"Ethanol": 100}, "T": 350}],
        "units": [{
            "id": "R1", "type": "reactor", "ins": ["f"], "outs": ["o"],
            "reaction": "Ethanol -> Acetaldehyde + Water",
            "reactant": "Ethanol", "conversion": 0.4, "T": 350,
        }],
    }
    out = builder.build(spec)
    rxn = next(c for c in out["verification"]["checks"]
               if c["name"] == "Reaction mass balance")
    assert rxn["status"] == "warn"
    assert out["verification"]["overall"] == "warn"


def test_multi_unit_process_with_mixer_and_splitter(builder):
    spec = {
        "name": "mix-react-split",
        "chemicals": ["Ethanol", "AceticAcid", "EthylAcetate", "Water"],
        "feeds": [
            {"name": "a", "flows": {"Ethanol": 100}, "T": 350},
            {"name": "b", "flows": {"AceticAcid": 100}, "T": 350},
        ],
        "units": [
            {"id": "M1", "type": "mixer", "ins": ["a", "b"], "outs": ["mixed"]},
            {"id": "R1", "type": "reactor", "ins": ["mixed"], "outs": ["crude"],
             "reaction": "Ethanol + AceticAcid -> EthylAcetate + Water",
             "reactant": "Ethanol", "conversion": 0.5, "T": 350},
            {"id": "S1", "type": "splitter", "ins": ["crude"],
             "outs": ["cut", "product"], "split": 0.3},
        ],
    }
    out = builder.build(spec)
    assert out["verification"]["overall"] == "pass"
    assert {u["id"] for u in out["results"]["units"]} == {"M1", "R1", "S1"}


@pytest.mark.parametrize("spec,msg", [
    ({"chemicals": ["Glucose"], "feeds": [{"name": "f", "flows": {"Glucose": 1}}],
      "units": [{"id": "M", "type": "mixer", "ins": ["f"], "outs": ["o"]}]},
     "allowlist"),
    ({"chemicals": ["Water"], "feeds": [{"name": "f", "flows": {"Water": 1}}],
      "units": [{"id": "F", "type": "flash", "ins": ["nope"], "outs": ["v", "l"], "V": 0.5}]},
     "before it is produced"),
    ({"chemicals": ["Water"], "feeds": [{"name": "f", "flows": {"Water": 1}}],
      "units": [{"id": "M", "type": "mixer", "ins": ["f"], "outs": ["o"]}]},
     ">= 2 inputs"),
    ({"chemicals": ["Ethanol", "Water"], "feeds": [{"name": "f", "flows": {"Ethanol": 1}}],
      "units": [{"id": "R", "type": "reactor", "ins": ["f"], "outs": ["o"],
                 "reaction": "Ethanol -> Foo", "reactant": "Ethanol", "conversion": 0.5}]},
     "not in this process"),
    ({"chemicals": ["Ethanol"], "feeds": [{"name": "f", "flows": {"Ethanol": 1}}],
      "units": [{"id": "R", "type": "reactor", "ins": ["f"], "outs": ["o"],
                 "reaction": "Ethanol -> Ethanol", "reactant": "Ethanol", "conversion": 5.0}]},
     "conversion must be in"),
])
def test_invalid_specs_rejected(builder, spec, msg):
    with pytest.raises(BuilderError) as exc:
        builder.build(spec)
    assert msg in str(exc.value)


def test_empty_spec_rejected(builder):
    with pytest.raises(BuilderError):
        builder.build({"chemicals": [], "feeds": [], "units": []})


# -- economics (A.1) ------------------------------------------------------

def _esterification_with_price(product="vapor", economics=None):
    spec = {
        "name": "ea with economics",
        "chemicals": ["Ethanol", "AceticAcid", "EthylAcetate", "Water"],
        "feeds": [{"name": "feed",
                   "flows": {"Ethanol": 100, "AceticAcid": 100},
                   "T": 350, "price": 0.5}],
        "units": [
            {"id": "R1", "type": "reactor", "ins": ["feed"], "outs": ["crude"],
             "reaction": "Ethanol + AceticAcid -> EthylAcetate + Water",
             "reactant": "Ethanol", "conversion": 0.6, "T": 350},
            {"id": "F1", "type": "flash", "ins": ["crude"],
             "outs": ["vapor", "liquid"], "V": 0.5},
        ],
        "product": product,
    }
    if economics is not None:
        spec["economics"] = economics
    return spec


def test_msp_computed_when_product_named(builder):
    out = builder.build(_esterification_with_price())
    econ = out["results"]["economics"]
    assert econ["product"] == "vapor"
    assert econ["min_selling_price_usd_per_kg"] > 0
    assert econ["total_capital_investment_usd"] > 0
    # feedstock-dominated: MSP should be within a sane band, not absurd.
    assert 0 < econ["min_selling_price_usd_per_kg"] < 100


def test_economic_check_added_to_verification(builder):
    out = builder.build(_esterification_with_price())
    names = {c["name"] for c in out["verification"]["checks"]}
    assert "Economic plausibility" in names
    assert out["verification"]["overall"] == "pass"


def test_no_economics_without_product(builder):
    spec = _esterification_with_price()
    del spec["product"]
    out = builder.build(spec)
    assert "economics" not in out["results"]
    names = {c["name"] for c in out["verification"]["checks"]}
    assert "Economic plausibility" not in names


def test_economics_overrides_applied(builder):
    out = builder.build(_esterification_with_price(economics={"IRR": 0.2, "plant_years": 15}))
    a = out["results"]["economics"]["assumptions"]
    assert a["IRR"] == 0.2
    assert a["plant_years"] == 15


def test_product_must_be_terminal(builder):
    with pytest.raises(BuilderError) as exc:
        builder.build(_esterification_with_price(product="crude"))
    assert "terminal product" in str(exc.value)


def test_unknown_economics_key_rejected(builder):
    with pytest.raises(BuilderError) as exc:
        builder.build(_esterification_with_price(economics={"bogus": 1}))
    assert "Unknown economics key" in str(exc.value)


def test_out_of_range_economics_rejected(builder):
    with pytest.raises(BuilderError) as exc:
        builder.build(_esterification_with_price(economics={"IRR": 5.0}))
    assert "outside allowed range" in str(exc.value)


def test_negative_feed_price_rejected(builder):
    spec = _esterification_with_price()
    spec["feeds"][0]["price"] = -1
    with pytest.raises(BuilderError) as exc:
        builder.build(spec)
    assert "price must be >= 0" in str(exc.value)


# -- recycles (A.2) -------------------------------------------------------

def _esterification_with_recycle(recycle_split=0.4):
    return {
        "name": "esterification with recycle",
        "chemicals": ["Ethanol", "AceticAcid", "EthylAcetate", "Water"],
        "recycles": ["recycle"],
        "feeds": [{"name": "feed",
                   "flows": {"Ethanol": 100, "AceticAcid": 100},
                   "T": 350, "price": 0.5}],
        "units": [
            {"id": "M1", "type": "mixer", "ins": ["feed", "recycle"], "outs": ["mixed"]},
            {"id": "R1", "type": "reactor", "ins": ["mixed"], "outs": ["crude"],
             "reaction": "Ethanol + AceticAcid -> EthylAcetate + Water",
             "reactant": "Ethanol", "conversion": 0.5, "T": 350},
            {"id": "F1", "type": "flash", "ins": ["crude"],
             "outs": ["vapor", "liquid"], "V": 0.5},
            {"id": "S1", "type": "splitter", "ins": ["liquid"],
             "outs": ["recycle", "product"], "split": recycle_split},
        ],
        "product": "vapor",
    }


def test_recycle_loop_converges_and_verifies(builder):
    out = builder.build(_esterification_with_recycle())
    assert out["verification"]["overall"] == "pass"
    # recycle is internal: both produced and consumed -> intermediate, not product
    assert "recycle" in {s["name"] for s in out["results"]["intermediates"]}
    assert "recycle" not in {s["name"] for s in out["results"]["products"]}


def test_recycle_carries_nonzero_flow(builder):
    builder.build(_esterification_with_recycle())
    recycle = next(s for s in builder.last_results["intermediates"]
                   if s["name"] == "recycle")
    assert recycle["F_mass_kg_hr"] > 0


def test_recycle_must_be_produced(builder):
    spec = _esterification_with_recycle()
    spec["recycles"] = ["recycle", "ghost"]
    with pytest.raises(BuilderError) as exc:
        builder.build(spec)
    assert "never produced" in str(exc.value)


def test_recycle_must_be_consumed(builder):
    spec = _esterification_with_recycle()
    # Add an orphan recycle that some unit produces but nobody consumes.
    spec["recycles"] = ["recycle", "orphan"]
    spec["units"][2]["outs"] = ["vapor", "liquid"]
    spec["units"].append(
        {"id": "S2", "type": "splitter", "ins": ["vapor"],
         "outs": ["orphan", "vent"], "split": 0.1}
    )
    spec["product"] = "vent"
    with pytest.raises(BuilderError) as exc:
        builder.build(spec)
    assert "never consumed" in str(exc.value)


def test_recycle_not_produced_twice(builder):
    spec = _esterification_with_recycle()
    spec["units"][2]["outs"] = ["recycle", "liquid"]  # flash also emits recycle
    with pytest.raises(BuilderError) as exc:
        builder.build(spec)
    assert "more than one unit" in str(exc.value)


def test_feed_cannot_be_recycle(builder):
    spec = _esterification_with_recycle()
    spec["recycles"] = ["recycle", "feed"]
    with pytest.raises(BuilderError) as exc:
        builder.build(spec)
    assert "both a feed and a recycle" in str(exc.value)


def test_recycle_improves_conversion_economics(builder):
    """Recycling unreacted feed should not raise the minimum selling price
    versus discarding it (more product recovered per unit feed)."""
    with_recycle = builder.build(_esterification_with_recycle())["results"]
    b2 = ProcessBuilder()
    no_recycle = b2.build(_esterification_with_price())["results"]
    assert (with_recycle["economics"]["min_selling_price_usd_per_kg"]
            <= no_recycle["economics"]["min_selling_price_usd_per_kg"])


# -- carbon (direct GHG) --------------------------------------------------

DECARBOXYLATION = {
    "name": "acetic acid decarboxylation",
    "chemicals": ["AceticAcid", "CH4", "CO2", "Water"],
    "feeds": [{"name": "feed", "flows": {"AceticAcid": 100}, "T": 400, "price": 0.4}],
    "units": [
        {"id": "R1", "type": "reactor", "ins": ["feed"], "outs": ["crude"],
         "reaction": "AceticAcid -> CH4 + CO2", "reactant": "AceticAcid",
         "conversion": 0.8, "T": 400},
        {"id": "S1", "type": "splitter", "ins": ["crude"],
         "outs": ["gas", "liquid"], "split": 0.9},
    ],
    "product": "liquid",
}


def test_carbon_present_and_scoped(builder):
    out = builder.build(ESTERIFICATION)
    carbon = out["results"]["carbon"]
    assert "not full LCA" in carbon["scope"]
    assert carbon["gwp_basis"] == "IPCC AR5 GWP100"


def test_carbon_zero_when_no_ghg(builder):
    out = builder.build(ESTERIFICATION)
    assert out["results"]["carbon"]["co2e_kg_per_hr"] == 0
    assert out["results"]["carbon"]["direct_ghg_kg_per_hr"] == {}


def test_carbon_counts_co2_and_ch4(builder):
    out = builder.build(DECARBOXYLATION)
    carbon = out["results"]["carbon"]
    ghg = carbon["direct_ghg_kg_per_hr"]
    assert ghg["CO2"] > 0 and ghg["CH4"] > 0
    # CO2e = CO2*1 + CH4*28
    expected = ghg["CO2"] * 1.0 + ghg["CH4"] * 28.0
    assert carbon["co2e_kg_per_hr"] == pytest.approx(expected, rel=1e-3)


def test_carbon_intensity_per_product(builder):
    out = builder.build(DECARBOXYLATION)
    carbon = out["results"]["carbon"]
    assert carbon["co2e_kg_per_kg_product"] > 0
    assert carbon["product"] == "liquid"
