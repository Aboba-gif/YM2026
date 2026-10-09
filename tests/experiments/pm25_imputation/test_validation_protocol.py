"""Скрытие ответов, признаки и применение результата заполнения PM₂.₅."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from timeseries.features import FeatureSet
from timeseries.imputation import input_layout

from experiments.pm25_imputation.data import SITES
from experiments.pm25_imputation.export import DEFAULT_PROTOCOL, evaluate, make_bank
from experiments.pm25_imputation.features import common_features, network_features
from experiments.pm25_imputation.imputation import PM25Imputer, pm25_imputer
from experiments.pm25_imputation.panel import evaluation_inputs, network_panel, visible_inputs


CONFIG = {"hgb": {"max_iter": 4, "min_samples_leaf": 3}, "common_window_steps": 64,
          "common_stride": 32, "common_lengths": [1, 3, 12, 36], "common_max_windows": 3}


@pytest.fixture(autouse=True)
def fixed_threads():
    with threadpool_limits(limits=1):
        yield


def panel():
    rng = np.random.default_rng(4)
    index = pd.date_range("2021-01-01", periods=300, freq="20min")
    return network_panel(index, rng.uniform(1, 40, (len(index), 4)))


def fit(original):
    train = np.ones(len(original.times), dtype=bool)
    inputs = visible_inputs(original, protocol_id="test", population_id="train", population_kind="train")
    return pm25_imputer(CONFIG, SITES[0]).fit(inputs, rng=np.random.default_rng(8), train_mask=train)


@pytest.mark.parametrize("geometry", ["internal", "leading", "trailing"])
@pytest.mark.parametrize("regime", ["actual", "none"])
def test_hidden_answers_do_not_enter_features_or_predictions(geometry, regime):
    original = panel()
    changed = original.values[:, :, 0].copy()
    changed[94:130, 0] += 1e9
    changed = network_panel(pd.DatetimeIndex(original.times), changed)
    row = SimpleNamespace(window_start=40, window_end=184, start=94, end=130,
                          geometry=geometry, site=SITES[0], split="test", block_id="block")
    inputs, truth, a, b = evaluation_inputs(original, row, regime, protocol_id="test")
    other, other_truth, _, _ = evaluation_inputs(changed, row, regime, protocol_id="test")
    np.testing.assert_array_equal(inputs.values, other.values)
    assert not np.array_equal(truth.values[truth.target_mask], other_truth.values[other_truth.target_mask])
    assert truth.target_mask.sum() == 36
    assert inputs.artificial_mask.sum() >= truth.target_mask.sum()
    np.testing.assert_array_equal(network_features(inputs, 0).values, network_features(other, 0).values)
    if geometry == "internal":
        np.testing.assert_array_equal(common_features(inputs, a, b, 0).values,
                                      common_features(other, a, b, 0).values)
    state = fit(original)
    for method in ("linear", "common_hgb", "network_hgb", "climatology"):
        imputer = pm25_imputer(CONFIG, SITES[0], method=method, gap=[a, b])
        first = imputer.predict(inputs, fitted=state)
        second = imputer.predict(other, fitted=state)
        first.require_binding(truth)
        np.testing.assert_array_equal(first.prediction, second.prediction)
        np.testing.assert_array_equal(first.availability, second.availability)
        np.testing.assert_array_equal(first.prediction[inputs.visible], inputs.values[inputs.visible])


def test_feature_values_are_used_in_model_training(monkeypatch):
    original = panel()
    transform = FeatureSet.transform

    def changed_features(features, inputs, *, context="retrospective"):
        result = transform(features, inputs, context=context)
        if len(features.names) == 13:
            values = result.values.copy()
            values[:, 0, 0] = 77.
            result = replace(result, values=values)
        return result

    from sklearn.ensemble import HistGradientBoostingRegressor
    fit_estimator = HistGradientBoostingRegressor.fit
    matrices = []

    def record_fit(estimator, x, y, *args, **kwargs):
        matrices.append(x.copy())
        return fit_estimator(estimator, x, y, *args, **kwargs)

    monkeypatch.setattr(FeatureSet, "transform", changed_features)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", record_fit)
    fit(original)
    np.testing.assert_array_equal(matrices[0][:len(original.times), 0], 77.)
    assert np.isnan(matrices[0][len(original.times):, 0]).all()


def test_result_prediction_is_used_for_scoring(monkeypatch):
    original = panel()
    state = fit(original)
    bank = pd.DataFrame([dict(block_id="block", site=SITES[0], split="test", geometry="internal",
                              length=3, window_start=40, window_end=184, start=100, end=103)])
    predict = PM25Imputer.predict

    def changed_prediction(imputer, inputs, *, fitted):
        result = predict(imputer, inputs, fitted=fitted)
        if imputer.method == "network_hgb":
            values = result.prediction.copy()
            values[imputer.gap[0]:imputer.gap[1], 0, 0] = 5.
            result = replace(result, prediction=values)
        return result

    monkeypatch.setattr(PM25Imputer, "predict", changed_prediction)
    metrics = evaluate(original, bank, {SITES[0]: state},
                       {"protocol_id": "test", "bank": {"neighbor_regimes": ["actual"]}})
    row = metrics[metrics.method.eq("network_hgb")].iloc[0]
    error = 5. - original.values[100:103, 0, 0]
    assert row.n == 3 and row.squared_sum == float(error @ error)


def test_fit_rejects_visible_values_outside_training_mask():
    original = panel()
    inputs = visible_inputs(original, protocol_id="test", population_id="train", population_kind="train")
    train = np.arange(len(original.times)) < 200
    with pytest.raises(ValueError, match="outside train_mask"):
        pm25_imputer(CONFIG, SITES[0]).fit(inputs, rng=np.random.default_rng(2), train_mask=train)


def test_prediction_rejects_different_layout_and_target():
    original = panel()
    state = fit(original)
    values = original.values[:, :, 0].copy()
    values[100:103, :] = np.nan
    inputs = visible_inputs(network_panel(pd.DatetimeIndex(original.times), values), protocol_id="test",
                            population_id="prediction", population_kind="export")
    assert state.layout == input_layout(inputs)
    with pytest.raises(ValueError, match="differ"):
        pm25_imputer(CONFIG, SITES[1], gap=[100, 103]).predict(inputs, fitted=state)
    shifted = network_panel(pd.DatetimeIndex(original.times) + pd.Timedelta("20min"), values)
    other = visible_inputs(shifted, protocol_id="test", population_id="prediction", population_kind="export")
    with pytest.raises(ValueError, match="differ"):
        pm25_imputer(CONFIG, SITES[0], gap=[100, 103]).predict(other, fitted=state)


def test_bank_replays_deterministically_within_declared_splits():
    import json
    cfg = deepcopy(json.loads(DEFAULT_PROTOCOL.read_bytes()))
    index = pd.date_range("2042-01-01", periods=6000, freq="20min")
    values = np.ones((len(index), 4))
    cfg["splits"] = {"selection": [str(index[1000]), str(index[3500])],
                     "test": [str(index[3500]), str(index[-1] + pd.Timedelta("20min"))]}
    cfg["bank"]["requested"] = {"selection": 2, "test": 2}
    cfg["bank"]["boundary_requested"] = {"selection": 1, "test": 1}
    first, counts = make_bank(index, values, cfg)
    second, second_counts = make_bank(index, values, cfg)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(counts, second_counts)
    for name, group in first.groupby("split"):
        start, end = cfg["splits"][name]
        assert (index[group.window_start] >= start).all()
        assert (index[group.window_end - 1] < end).all()
