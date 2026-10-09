"""Условия входов модели для заполнения всего ряда и границы обучения."""
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from experiments.pm25_imputation.models import CoreModels, reorder
from experiments.pm25_imputation.features import network_features
from experiments.pm25_imputation.panel import network_panel, visible_inputs


CONFIG = {
    "hgb": {"max_iter": 4, "min_samples_leaf": 3},
    "common_window_steps": 32,
    "common_stride": 8,
    "common_lengths": [1, 3, 12],
    "common_max_windows": 6,
}


def panel(n=240):
    index = pd.date_range("2020-01-01", periods=n, freq="20min")
    rng = np.random.default_rng(501)
    values = rng.uniform(1, 20, (n, 4))
    values[:, 0] = 2 * values[:, 1] + values[:, 2] / 3
    return index, values


def inputs(index, values, train=None):
    return visible_inputs(network_panel(index, values), protocol_id="test",
                          population_id="train" if train is not None else "prediction",
                          population_kind="train" if train is not None else "export",
                          train_mask=train)


@pytest.fixture(autouse=True)
def fixed_threads():
    with threadpool_limits(limits=1):
        yield


def test_reorder_copies_and_network_features_ignore_entire_target_column():
    index, values = panel()
    permuted = reorder(values, 2)
    np.testing.assert_array_equal(permuted[:, 0], values[:, 2])
    np.testing.assert_array_equal(permuted[:, 1:], values[:, [0, 1, 3]])
    permuted[0, 0] = -999
    assert values[0, 2] > 0
    changed = values.copy()
    changed[:, 0] = np.where(np.arange(len(values)) % 2, 1e12, np.nan)
    np.testing.assert_array_equal(network_features(inputs(index, values), 0).values, network_features(inputs(index, changed), 0).values)
    with pytest.raises(ValueError, match="four-station"):
        reorder(values[:, :3], 0)


def test_fit_excludes_all_outside_training_values_and_windows_cannot_cross_holes():
    index, values = panel()
    train = np.arange(len(index)) < 168
    train[64:96] = False
    first = CoreModels(CONFIG).fit(inputs(index, values, train), train)
    changed = values.copy()
    changed[~train] *= 1000
    second = CoreModels(CONFIG).fit(inputs(index, changed, train), train)
    assert first.fit_record == second.fit_record
    for start in first.fit_record["common_selected_window_starts"]:
        assert train[start:start + CONFIG["common_window_steps"]].all()
    visible = values.copy()
    visible[181:184, 0] = np.nan
    a = first.predict_gap(inputs(index, visible), 181, 184)
    b = second.predict_gap(inputs(index, visible), 181, 184)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_gap_concealment_no_input_mutation_and_long_edge_fallback():
    index, values = panel()
    values[30:40, 1:] = np.nan
    model = CoreModels(CONFIG).fit(inputs(index, values, np.arange(len(index)) < 170), np.arange(len(index)) < 170)
    with pytest.raises(ValueError, match="Conceal every"):
        model.predict_gap(inputs(index, values), 180, 183)
    for a, b, bracketed, common in [(180, 183, True, True), (0, 30, False, False),
                                    (200, 240, False, False), (175, 195, True, False)]:
        visible = values.copy()
        visible[a:b, 0] = np.nan
        before = visible.copy()
        result = model.predict_gap(inputs(index, visible), a, b)
        np.testing.assert_array_equal(visible, before)
        assert (result["linear"] is not None) == bracketed
        assert (result["common_hgb"] is not None) == common
        assert np.isfinite(result["network_hgb"]).all()
        assert np.isfinite(result["climatology"]).all()
        if bracketed:
            expected = (1 - np.arange(1, b - a + 1) / (b - a + 1)) * values[a - 1, 0]
            expected += np.arange(1, b - a + 1) / (b - a + 1) * values[b, 0]
            np.testing.assert_allclose(result["linear"], expected)


def test_sparse_training_has_finite_fallback_without_common_model():
    index, values = panel(64)
    values[:, 0] = np.nan
    values[[1, 5], 0] = [2.0, 8.0]
    values[:, 1:] = np.nan
    model = CoreModels(CONFIG).fit(inputs(index, values, np.ones(len(index), dtype=bool)), np.ones(len(index), dtype=bool))
    assert not model.fit_record["common_available"]
    assert model.fit_record["network_original_training_labels"] == 2
    assert model.fit_record["climatology_global_mean"] == 5
    result = model.predict_gap(inputs(index, values), 10, 40)
    assert result["common_hgb"] is None
    np.testing.assert_array_equal(result["climatology"], np.full(30, 5.0))
    assert np.isfinite(result["network_hgb"]).all()
    with pytest.raises(ValueError, match="originally observed"):
        CoreModels(CONFIG).fit(inputs(index, values, np.zeros(len(index), dtype=bool)), np.zeros(len(index), dtype=bool))


def test_common_training_cap_and_counts_are_deterministic_and_original_only():
    index, values = panel(300)
    values[80:100, 0] = np.nan
    original = values.copy()
    model = CoreModels(CONFIG).fit(inputs(index, values, np.ones(len(index), dtype=bool)), np.ones(len(index), dtype=bool))
    record = model.fit_record
    assert len(record["common_selected_window_starts"]) == 6
    assert record["common_eligible_windows"] > 6
    assert record["common_unique_original_targets"] <= np.isfinite(values[:, 0]).sum()
    assert record["common_augmented_target_occurrences"] == sum(
        2 * row["length"] for row in record["common_training_support"])
    np.testing.assert_array_equal(values, original)
    for row in record["common_training_support"]:
        a = row["window_start"] + (CONFIG["common_window_steps"] - row["length"]) // 2
        b = a + row["length"]
        assert np.isfinite(values[a - 1:b + 1, 0]).all()


def test_invalid_raw_values_and_irregular_clock_are_rejected():
    index, values = panel()
    train = np.ones(len(index), dtype=bool)
    wrong = values.copy()
    wrong[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        CoreModels(CONFIG).fit(inputs(index, wrong, train), train)
    with pytest.raises(ValueError, match="20-minute"):
        CoreModels(CONFIG).fit(inputs(index.delete(5), np.delete(values, 5, axis=0), np.delete(train, 5)), np.delete(train, 5))
