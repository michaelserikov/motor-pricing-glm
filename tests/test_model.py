"""Offline tests on synthetic data: features, metrics, large-claim load and the aggregate simulation."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from motorpricing import model as m


def test_bin_features_bands_and_caps():
    x = pd.DataFrame({"DrivAge": [18, 45, 80], "VehAge": [0, 3, 30], "VehPower": [4, 9, 15], "BonusMalus": [50, 100, 230],
                      "Density": [1, 100, 27000], "Area": ["A", "B", "C"], "VehBrand": ["B1", "B2", "B3"],
                      "VehGas": ["Diesel", "Regular", "Diesel"], "Region": ["R11", "R24", "R82"]})
    f = m.bin_features(x)
    assert f["DrivAgeBand"].tolist() == ["(17, 20]", "(40, 50]", "(70, 120]"]
    assert f["VehAgeBand"].tolist() == ["(-1, 0]", "(1, 5]", "(15, 120]"]
    assert f["VehPowerCap"].tolist() == ["4", "9", "9"]
    assert f["BonusMalus"].max() == 150
    assert f["LogDensity"].iloc[0] == 0.0


def test_gini_is_zero_for_constant_and_positive_for_a_good_ranking():
    rng = np.random.default_rng(0)
    risk = rng.gamma(2.0, 0.05, 20_000)
    y = rng.poisson(risk).astype(float)
    w = np.ones_like(y)
    assert m.lorenz_gini(y, np.full_like(y, 0.1), w) == pytest.approx(0.0, abs=0.02)
    assert m.lorenz_gini(y, risk, w) > 0.2
    assert m.lorenz_gini(y, -risk, w) < -0.2


def test_calibration_table_preserves_totals():
    rng = np.random.default_rng(1)
    w = rng.uniform(0.1, 1.0, 1000)
    p = rng.uniform(0.05, 0.2, 1000)
    y = rng.poisson(p * w) / w
    t = m.calibration_table(y, p, w, groups=5)
    assert len(t) == 5
    assert (t["observed_rate"] * t["exposure"]).sum() == pytest.approx((y * w).sum())
    assert t["predicted_rate"].is_monotonic_increasing


def test_double_lift_is_relative_to_each_portfolio_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a, b = 10 * y, np.full(4, 5.0)
    d = m.double_lift(y, a, b, np.ones(4), groups=2)
    assert d["model_a"].tolist() == pytest.approx(d["observed"].tolist())
    assert d["model_b"].tolist() == pytest.approx([1.0, 1.0])


def test_gpd_load_recovers_shape_and_empirical_load():
    rng = np.random.default_rng(2)
    claims = np.concatenate([rng.uniform(0, 1000, 9000), 1000 + genpareto.rvs(0.3, scale=500, size=1000, random_state=rng)])
    g = m.gpd_load(claims, 1000.0)
    assert g["xi"] == pytest.approx(0.3, abs=0.1)
    assert g["p_above"] == pytest.approx(0.1)
    assert g["load_per_claim_empirical"] == pytest.approx(np.clip(claims - 1000, 0, None).mean())


def test_retained_excess_matches_the_exponential_closed_form():
    g = {"threshold": 100.0, "xi": 0.0, "sigma": 50.0}
    assert m.retained_excess_gpd(g, 200.0) == pytest.approx(50 * (1 - np.exp(-2)), rel=1e-6)
    assert m.retained_excess_gpd(g, 50.0) == 0.0


def test_simulated_aggregate_mean_with_retention():
    body = np.array([100.0, 300.0])
    g = {"threshold": 400.0, "p_above": 0.1, "xi": 0.0, "sigma": 200.0}
    agg = m.simulate_aggregate(50.0, body, g, years=20_000, seed=3, retention=600.0)
    expected = 50 * (0.9 * 200 + 0.1 * (400 + 200 * (1 - np.exp(-1))))
    assert agg.mean() == pytest.approx(expected, rel=0.01)
    mixed = m.simulate_aggregate(50.0, body, g, years=20_000, cv_systematic=0.2, seed=3, retention=600.0)
    assert mixed.std() > agg.std()
