"""Проверки минимума Тихонова, инвариантов сглаживания и условий входов."""

import numpy as np
import pytest
from scipy.optimize import minimize

from experiments.pm25_imputation.smoothing import smooth_pm25, smoothing_diagnostics


def test_minimizer_matches_independent_constrained_optimization():
    y = np.array([0.0, 9.0, 1.0, 0.0, 6.0, 2.0])
    weights = np.array([1.0, 0.2, 1.0, 0.1, 1.0, 0.4])
    lam = 1.7
    expected = minimize(
        lambda z: 0.5 * np.sum(weights * (z - y) ** 2)
        + 0.5 * lam * np.sum(np.diff(z) ** 2),
        y,
        bounds=[(0, None)] * len(y),
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-7},
    )
    assert expected.success
    actual = smooth_pm25(y, weights=weights, lam=lam)
    np.testing.assert_allclose(actual, expected.x, rtol=2e-6, atol=2e-6)
    diagnostics = smoothing_diagnostics(y, actual, weights=weights, lam=lam)
    assert diagnostics["normal_equation_relative_residual"] < 1e-14
    assert diagnostics["objective"] < diagnostics["objective_at_input"]
    assert diagnostics["roughness_after"] < diagnostics["roughness_before"]


def test_maximum_principle_constant_and_weighted_mass():
    rng = np.random.default_rng(827)
    y = rng.uniform(0.0, 80.0, 713)
    weights = rng.choice([0.1, 1.0], len(y))
    z = smooth_pm25(y, weights=weights, lam=4.0)
    assert z.min() >= y.min()
    assert z.max() <= y.max()
    np.testing.assert_allclose(weights @ z, weights @ y, rtol=1e-14)
    np.testing.assert_allclose(
        smooth_pm25(np.full(len(y), 17.0), weights=weights, lam=4.0),
        17.0,
        rtol=1e-14,
    )
    np.testing.assert_array_equal(smooth_pm25(np.zeros(12), lam=50), 0.0)


def test_identity_singleton_and_no_input_mutation():
    y = np.array([0.0, 3.0, 7.0, 2.0])
    before = y.copy()
    z = smooth_pm25(y, weights=[1.0, 0.1, 0.1, 1.0], lam=0)
    np.testing.assert_array_equal(z, y)
    assert not np.shares_memory(y, z)
    smooth_pm25(y, lam=1.0)
    np.testing.assert_array_equal(y, before)
    np.testing.assert_array_equal(smooth_pm25([2.0], lam=12345), [2.0])


def test_full_four_year_grid_is_finite_and_stationary():
    t = np.arange(105192, dtype=float)
    y = 15 + 10 * np.sin(t / 20) + 3 * (t % 2)
    weights = np.where((t.astype(int) % 23) < 4, 0.1, 1.0)
    z = smooth_pm25(y, weights=weights, lam=2.0)
    diagnostics = smoothing_diagnostics(y, z, weights=weights, lam=2.0)
    assert z.shape == y.shape
    assert diagnostics["normal_equation_relative_residual"] < 1e-14
    assert diagnostics["objective"] <= diagnostics["objective_at_input"]
    assert np.all(np.isfinite(z)) and np.all(z >= 0)


def test_invalid_inputs_are_rejected():
    for values in ([], [[1, 2]], [0, np.nan], [np.inf, 0], [-1, 0]):
        with pytest.raises(ValueError):
            smooth_pm25(values)
    for weights in ([0, 1], [-1, 1], [np.nan, 1], [1], [[1, 1]]):
        with pytest.raises(ValueError):
            smooth_pm25([1, 2], weights=weights)
    for lam in (-1, np.nan, np.inf, [1]):
        with pytest.raises(ValueError):
            smooth_pm25([1, 2], lam=lam)
    with pytest.raises(ValueError):
        smoothing_diagnostics([1, 2], [1, np.nan])
