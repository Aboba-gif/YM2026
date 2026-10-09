"""Проверки метрики RMSE из публичного интерфейса для выбранного прогноза КрАЗ."""
import json

import numpy as np
import pandas as pd
import pytest

from timeseries.metrics import Metric, build_metrics
from experiments.kraz_alignment import run as kraz


@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
def test_public_rmse_matches_existing_scalar_float64_arithmetic(layout):
    target = np.array([1e8+0.123, 0.23, -5.171], dtype=np.float64)
    prediction = np.array([1e8-0.321, 0.765, -4.444], dtype=np.float64)
    if layout == "strided":
        target, prediction = np.repeat(target, 2)[::2], np.repeat(prediction, 2)[::2]
    elif layout == "reversed":
        target, prediction = target[::-1], prediction[::-1]
    expected = float(np.sqrt(np.mean((prediction-target)**2)))
    metrics = build_metrics([{"kind": "rmse", "parameters": {}}])
    assert metrics.score(target, prediction)["rmse"] == expected


def test_selected_withheld_error_saves_the_actual_public_metric_result(
        short_inputs, monkeypatch):
    config, _, output = short_inputs
    calls = []
    original_score, original_save = Metric.score, kraz.save_json

    def score(metric, target, prediction):
        result = original_score(metric, target, prediction)
        calls.append(dict(result=result, target=np.array(target, copy=True),
                          prediction=np.array(prediction, copy=True),
                          target_dtype=str(np.asarray(target).dtype),
                          prediction_dtype=str(np.asarray(prediction).dtype),
                          target_shape=np.asarray(target).shape,
                          prediction_shape=np.asarray(prediction).shape,
                          metric_module=type(metric).__module__, metric_name=metric.name))
        return result

    def save(path, payload):
        assert len(calls) == len(payload["selected"]) == 2
        for record, call in zip(payload["selected"], calls):
            assert record["unscaled_holdout_rmse"] is call["result"]
        return original_save(path, payload)

    monkeypatch.setattr(Metric, "score", score)
    monkeypatch.setattr(kraz, "save_json", save)
    kraz.run(config)
    manifest = json.loads((output/"manifest.json").read_bytes())
    overlay = pd.read_csv(output/"overlays.csv", float_precision="round_trip")
    for record, call in zip(manifest["selected"], calls):
        points = overlay[(overlay.source == record["source"])
                         & (overlay.split == record["split"]) & ~overlay.used_for_fit]
        assert len(points) == 3
        np.testing.assert_array_equal(call["target"], points.raw_pm25.to_numpy())
        np.testing.assert_array_equal(call["prediction"], points.unscaled_prediction.to_numpy())
        assert call["target_shape"] == call["prediction_shape"] == (3,)
        assert call["target_dtype"] == call["prediction_dtype"] == "float64"
        assert call["metric_module"] == "timeseries.metrics" and call["metric_name"] == "rmse"
        assert record["unscaled_holdout_rmse"] == call["result"]
