"""Regression tests: the assistant's tool layer must match raw BioSTEAM.

These tests pin the engine's numbers against values computed directly from
the biorefineries package, and verify safety behavior (bounds, reset).
They do not require an Anthropic API key.
"""
import math
import warnings

import pytest

from biosteam_ai.config import KG_ETHANOL_PER_GAL
from biosteam_ai.engine import SimulationSession, available_models


@pytest.fixture(scope="module")
def cornstover_session():
    s = SimulationSession()
    s.load_model("cornstover")
    s.run()
    return s


def _raw_cornstover_mesp_per_gal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from biorefineries import cornstover as cs

        cs.load()
        return cs.cornstover_tea.solve_price(cs.ethanol) * KG_ETHANOL_PER_GAL


from biosteam_ai.models import REGISTRY


def test_registry_lists_expected_models():
    keys = {m["key"] for m in available_models()}
    assert {"cornstover", "sugarcane", "lipidcane"} <= keys


@pytest.mark.parametrize("model_key", sorted(REGISTRY))
def test_every_model_loads_runs_and_resets(model_key):
    s = SimulationSession()
    s.load_model(model_key)
    metrics = s.run()
    mesp = metrics["mesp_per_gal"]["value"]
    assert mesp == mesp  # not NaN
    # changing feedstock price should move the price, then reset restores it.
    before = s.get_metrics()["mesp_per_gal"]["value"]
    s.set_parameter("feedstock_price", s.get_parameter("feedstock_price") * 1.5)
    s.run()
    s.reset_parameters()
    after = s.run()["mesp_per_gal"]["value"]
    assert math.isclose(before, after, rel_tol=3e-3)


def test_baseline_matches_raw_biosteam(cornstover_session):
    engine_value = cornstover_session.get_metrics()["mesp_per_gal"]["value"]
    raw_value = _raw_cornstover_mesp_per_gal()
    assert math.isclose(engine_value, raw_value, rel_tol=1e-3)


def test_mesp_in_reasonable_range(cornstover_session):
    mesp = cornstover_session.get_metrics()["mesp_per_gal"]["value"]
    assert 1.0 < mesp < 4.0


def test_lower_conversion_raises_price(cornstover_session):
    s = cornstover_session
    s.reset_parameters()
    baseline = s.run()["mesp_per_gal"]["value"]
    s.set_parameter("glucose_to_ethanol_conversion", 0.80)
    lowered = s.run()["mesp_per_gal"]["value"]
    assert lowered > baseline
    s.reset_parameters()
    s.run()


def test_out_of_bounds_rejected(cornstover_session):
    with pytest.raises(ValueError):
        cornstover_session.set_parameter("glucose_to_ethanol_conversion", 1.5)


def test_unknown_parameter_rejected(cornstover_session):
    with pytest.raises(KeyError):
        cornstover_session.set_parameter("does_not_exist", 1.0)


def test_reset_restores_baseline(cornstover_session):
    s = cornstover_session
    s.reset_parameters()
    baseline = s.run()["mesp_per_gal"]["value"]
    s.set_parameter("feedstock_price", 0.20)
    s.run()
    s.reset_parameters()
    restored = s.run()["mesp_per_gal"]["value"]
    assert math.isclose(baseline, restored, rel_tol=2e-3)


def test_sensitivity_sweep_monotonic_in_feedstock_price(cornstover_session):
    s = cornstover_session
    s.reset_parameters()
    result = s.sensitivity("feedstock_price", [0.04, 0.06, 0.08])
    prices = [row["metrics"]["mesp_per_gal"] for row in result["sweep"]]
    assert prices[0] < prices[1] < prices[2]


def test_sensitivity_stores_artifact(cornstover_session):
    s = cornstover_session
    s.sensitivity("feedstock_price", [0.04, 0.06])
    assert s.last_artifact is not None
    assert s.last_artifact["kind"] == "sensitivity"


def test_compare_scenarios_directions(cornstover_session):
    s = cornstover_session
    result = s.compare_scenarios(
        {
            "low_conversion": {"glucose_to_ethanol_conversion": 0.80},
            "cheap_feedstock": {"feedstock_price": 0.03},
        }
    )
    # Lower conversion should raise price; cheaper feedstock should lower it.
    assert result["scenarios"]["low_conversion"]["pct_change_vs_baseline"]["mesp_per_gal"] > 0
    assert result["scenarios"]["cheap_feedstock"]["pct_change_vs_baseline"]["mesp_per_gal"] < 0


def test_compare_scenarios_restores_baseline(cornstover_session):
    s = cornstover_session
    before = s.get_metrics()["mesp_per_gal"]["value"]
    s.compare_scenarios({"x": {"feedstock_price": 0.15}})
    after = s.get_metrics()["mesp_per_gal"]["value"]
    assert math.isclose(before, after, rel_tol=2e-3)


def test_uncertainty_summary_ordered_and_deterministic(cornstover_session):
    s = cornstover_session
    dists = [{"name": "feedstock_price", "low": 0.04, "high": 0.07}]
    r1 = s.uncertainty(dists, n_samples=40, seed=123)
    r2 = s.uncertainty(dists, n_samples=40, seed=123)
    m1 = r1["summary"]["mesp_per_gal"]
    assert m1["p5"] <= m1["p50"] <= m1["p95"]
    assert m1["min"] <= m1["mean"] <= m1["max"]
    # Same seed -> same parameter draws -> reproducible to solver tolerance.
    # (BioSTEAM's recycle solver warm-starts, so results are not bit-identical.)
    assert math.isclose(m1["mean"], r2["summary"]["mesp_per_gal"]["mean"], rel_tol=1e-3)
    assert r1["n_samples"] == 40


def test_uncertainty_rejects_out_of_bounds(cornstover_session):
    with pytest.raises(ValueError):
        cornstover_session.uncertainty(
            [{"name": "glucose_to_ethanol_conversion", "low": 0.5, "high": 1.5}],
            n_samples=5,
        )


def test_uncertainty_rejects_unknown_parameter(cornstover_session):
    with pytest.raises(KeyError):
        cornstover_session.uncertainty([{"name": "nope", "low": 0, "high": 1}], n_samples=5)
