"""Проверки экспорта, происхождения значений и выбора методов PM₂.₅."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.pm25_imputation.export import DEFAULT_PROTOCOL, aggregate, build_export, error_sums, evaluate, policy_choice, select_policy, verify_export
from experiments.pm25_imputation.data import SITES
from experiments.pm25_imputation.models import CoreModels
from timeseries.metrics import MetricSet
from timeseries.imputation import input_layout
from experiments.pm25_imputation.panel import network_panel, visible_inputs, evaluation_inputs
from experiments.pm25_imputation.imputation import PM25Fit


def policy():
    common = dict(method='network_hgb',selection_available=True,selection_blocks=8,selection_rmse=1.)
    return dict(internal=[dict(site=s,regime=r,length=n,**common) for s in SITES for r in ['actual','none'] for n in [1,3,12,36,72,216,720]],
                fallback=[dict(site=s,regime=r,**common) for s in SITES for r in ['actual','none']])


class FixedModel(CoreModels):
    """Тестовая модель с постоянными предсказаниями и проверкой скрытого блока."""

    def predict_gap(self, inputs, start, end, target_index=0, *, method=None):
        assert np.isnan(inputs.values[start:end, target_index, 0]).all()
        result = dict(linear=None, common_hgb=None, network_hgb=np.full(end-start, 7.),
                      climatology=np.full(end-start, 3.))
        return result if method is None else result[method]


def fixed_models(panel, config=None, sites=SITES):
    layout = input_layout(visible_inputs(panel, protocol_id='test', population_id='fixture', population_kind='export'))
    return {site: PM25Fit(FixedModel(config), site, layout) for site in sites}


def test_complete_serialization_and_extrapolation(tmp_path):
    n=1000
    idx=pd.date_range('2019-01-01',periods=n,freq='20min')
    values=np.full((n,4),2.)
    values[:10,0]=np.nan
    values[30:800,1]=np.nan
    values[900:,2]=np.nan
    values[20:23,3]=np.nan
    raw=values.copy()
    cfg=json.loads(DEFAULT_PROTOCOL.read_text())
    output,ledger,annual,smooth=build_export(network_panel(idx, values), fixed_models(network_panel(idx, values)), policy(), cfg)
    path=tmp_path/'final.csv'
    output.to_csv(path,index=False)
    record=verify_export(path,idx,raw)
    assert record['passed'] and record['columns']==33
    assert set(ledger.status)=={'leading_extrapolation','long_gap_extrapolation','trailing_extrapolation','artificial_mask_evaluated'}
    np.testing.assert_array_equal(values,raw)
    assert sum(annual.imputed)==883
    assert len(smooth)==16
    output.loc[0:9,'Severny_status']='artificial_mask_evaluated'
    output.to_csv(path,index=False)
    with pytest.raises(ValueError,match='mislabeled'):
        verify_export(path,idx,raw)


def test_verifier_rejects_imputed_as_observed(tmp_path):
    idx=pd.date_range('2020-01-01',periods=10,freq='20min')
    values=np.ones((10,4)); values[3:5,0]=np.nan
    cfg=json.loads(DEFAULT_PROTOCOL.read_text())
    out,*_=build_export(network_panel(idx, values), fixed_models(network_panel(idx, values)), policy(), cfg)
    path=tmp_path/'bad.csv'
    out.loc[3,'Severny_pm25_observed']=7
    out.to_csv(path,index=False)
    with pytest.raises(ValueError,match='original values'):
        verify_export(path,idx,values)


def test_concealment_destroys_target_context_and_neighbors():
    idx=pd.date_range('2021-01-01',periods=20,freq='20min')
    original=np.arange(80,dtype=float).reshape(20,4)
    row=SimpleNamespace(window_start=2,window_end=18,start=8,end=12,geometry='leading',
                        site=SITES[0], split='test', block_id='block')
    visible, truth, a, b = evaluation_inputs(network_panel(idx, original), row, 'none', protocol_id='test')
    assert np.isnan(visible.values[:b,0,0]).all() and np.isnan(visible.values[:,1:,0]).all()
    np.testing.assert_array_equal(truth.values[truth.target_mask],original[8:12,0])
    np.testing.assert_array_equal(original,np.arange(80).reshape(20,4))


def test_no_test_policy_selection_and_length_ceiling():
    cfg=json.loads(DEFAULT_PROTOCOL.read_text())
    bad=pd.DataFrame([dict(split='test')])
    with pytest.raises(ValueError,match='only selection'):
        select_policy(bad,cfg)
    assert policy_choice(policy(),'Severny',13,'actual')['length']==36
    assert 'length' not in policy_choice(policy(),'Severny',721,'actual')


def test_pooled_rmse_is_not_mean_block_rmse():
    rows=pd.DataFrame([dict(group='a',**error_sums(np.zeros(1),np.ones(1))),
                       dict(group='a',**error_sums(np.zeros(3),np.full(3,3.)))])
    got=aggregate(rows,['group']).iloc[0]
    assert got.rmse==pytest.approx(np.sqrt(7))
    assert got.n==4 and got.blocks==2


def test_error_sums_uses_public_metric_result(monkeypatch):
    """Использовать возвращённые API метрики, сохраняя число ответов."""
    truth = np.array([2., 3., 5.])
    prediction = np.array([4., 1., 6.])
    score = MetricSet.score
    calls = []

    def changed_score(metrics, target, predicted):
        calls.append((target, predicted))
        result = score(metrics, target, predicted)
        result['bias'] = -7.125
        return result

    monkeypatch.setattr(MetricSet, 'score', changed_score)
    result = error_sums(truth, prediction)
    assert len(calls) == 1
    assert calls[0][0] is truth and calls[0][1] is prediction
    assert result['bias'] == -7.125
    assert type(result['n']) is int and result['n'] == 3
    assert list(result) == ['n', 'squared_sum', 'absolute_sum', 'error_sum',
                            'rmse', 'mae', 'bias']


@pytest.mark.parametrize('truth,prediction', [
    (np.zeros(0), np.zeros(0)),
    (np.array([np.nan]), np.array([1.])),
    (np.array([1.]), np.array([np.inf])),
    (np.array([1. + 1.j]), np.array([2. + 2.j])),
    (np.array([0.]), np.array([1.e200])),
    (np.ones(2), np.ones(3)),
])
def test_error_sums_rejects_unscorable_blocks(truth, prediction):
    """Не записывать неполные или неконечные метрики проверочного блока."""
    with pytest.raises(ValueError):
        error_sums(truth, prediction)


@pytest.mark.parametrize('dtype', [str, object])
def test_error_sums_rejects_non_numeric_prediction_dtype(dtype):
    """Не приводить объектные и строковые прогнозы к числам неявно."""
    with pytest.raises(TypeError):
        error_sums(np.array([1.]), np.array([2.], dtype=dtype))


@pytest.mark.parametrize('geometry', ['internal', 'leading', 'trailing'])
@pytest.mark.parametrize('regime', ['actual', 'none'])
def test_evaluate_preserves_concealment_and_skips_unavailable_methods(geometry, regime):
    """Оценивать скрытый блок, не передавая модели его контрольные значения."""
    index = pd.date_range('2021-01-01', periods=20, freq='20min')
    values = np.arange(80., dtype=float).reshape(20, 4)
    original = values.copy()
    site = SITES[0]
    bank = pd.DataFrame([dict(block_id='block', site=site, split='test',
                              geometry=geometry, length=4, window_start=2,
                              window_end=18, start=8, end=12)])
    panel = network_panel(index, values)
    fitted = fixed_models(panel, sites=[site])[site]
    model = fitted.models
    predict_gap = model.predict_gap
    calls = []

    def predict(inputs, start, end, target_index=0, *, method=None):
        visible = inputs.values[:, :, 0]
        assert np.isnan(visible[start:end, 0]).all()
        assert np.isnan(visible[:start, 0]).all() == (geometry == 'leading')
        assert np.isnan(visible[end:, 0]).all() == (geometry == 'trailing')
        assert np.isnan(visible[:, 1:]).all() == (regime == 'none')
        calls.append((start, end, method))
        return predict_gap(inputs, start, end, target_index, method=method)

    model.predict_gap = predict
    metrics = evaluate(panel, bank, {site: fitted},
                       {'protocol_id': 'test', 'bank': {'neighbor_regimes': [regime]}})
    assert calls == [(6, 10, method) for method in ['network_hgb', 'climatology', 'linear', 'common_hgb']]
    assert metrics.method.tolist() == ['network_hgb', 'climatology']
    assert metrics.n.tolist() == [4, 4]
    np.testing.assert_array_equal(metrics.squared_sum, [3924., 4980.])
    np.testing.assert_array_equal(metrics.absolute_sum, [124., 140.])
    np.testing.assert_array_equal(metrics.error_sum, [-124., -140.])
    np.testing.assert_array_equal(metrics.rmse, np.sqrt([981., 1245.]))
    np.testing.assert_array_equal(metrics.mae, [31., 35.])
    np.testing.assert_array_equal(metrics.bias, [-31., -35.])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("width,lengths,neighbor_index,regime,method,fraction", [
    (288, [1, 3, 12, 36, 72, 216], 5, "none", "climatology", 0.0),
    (360, [1, 3, 12, 36, 72, 216], 5, "actual", "network_hgb", 1 / 1080),
    (288, [1, 3, 12], 40, "none", "climatology", 0.0),
    (288, [1, 3, 12, 36, 72, 216], 40, "actual", "network_hgb", 1 / 864),
])
def test_export_regime_uses_effective_common_window(
        width, lengths, neighbor_index, regime, method, fraction):
    """Выбирать политику по контексту окна из конфигурации модели."""
    n = 1000
    index = pd.date_range("2019-01-01", periods=n, freq="20min")
    values = np.full((n, 4), np.nan)
    values[:, 0] = 2.0
    values[150:186, 0] = np.nan
    values[neighbor_index, 1] = 2.0
    values[-1, 2:] = 2.0
    original = values.copy()
    config = {"common_window_steps": width, "common_lengths": lengths}
    panel = network_panel(index, values)
    models = fixed_models(panel, config)
    cfg = json.loads(DEFAULT_PROTOCOL.read_text())
    cfg["models"].update(config)
    chosen = policy()
    for row in chosen["internal"] + chosen["fallback"]:
        if row["regime"] == "none":
            row["method"] = "climatology"

    output, ledger, _, _ = build_export(panel, models, chosen, cfg)
    gap = ledger[(ledger.site == "Severny") & (ledger.start_index == 150)].iloc[0]
    assert gap.end_index_exclusive == 186
    assert gap.neighbor_regime == regime
    assert gap.method == method
    assert gap.observed_neighbor_context_fraction == pytest.approx(fraction)
    expected = 7.0 if method == "network_hgb" else 3.0
    np.testing.assert_array_equal(output["Severny_pm25_filled"][150:186], expected)
    np.testing.assert_array_equal(values, original)
    for j, site in enumerate(SITES):
        observed = np.isfinite(original[:, j])
        np.testing.assert_array_equal(output[f"{site}_pm25_observed"], original[:, j])
        np.testing.assert_array_equal(output[f"{site}_pm25_filled"][observed], original[observed, j])
        np.testing.assert_array_equal(output[f"{site}_is_imputed"], ~observed)
        np.testing.assert_array_equal(output[f"{site}_adr_observation_weight"], observed.astype(int))
