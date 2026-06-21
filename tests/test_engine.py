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


def test_registry_lists_expected_models():
    keys = {m["key"] for m in available_models()}
    assert {"cornstover", "sugarcane"} <= keys


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
