"""Проверки популяционной цели, структуры ковариаций, KL и численных ограничений."""
import json

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from experiments.observation_sensitivity import CovarianceGuards, ExponentialMixture, covariance_diagnostics, matched_lag_mixture, mixture_covariance, population_derivative, population_objective


LENGTH = .24109057879930182


def test_matched_lag_and_nonexponential_second_lag():
    mixture = matched_lag_mixture(1/3, LENGTH)
    af, ass = np.exp(-(1/3)/mixture.fast_hours), np.exp(-(1/3)/mixture.slow_hours)
    r1, r2 = mixture.correlation([1/3,2/3])
    assert r1 == pytest.approx(np.exp(-(1/3)/LENGTH), rel=1e-14)
    expected = mixture.fast_weight*(1-mixture.fast_weight)*(af-ass)**2
    assert r2-r1**2 == pytest.approx(expected, rel=1e-14)
    assert expected > 0


@pytest.mark.parametrize("n", [2,9,17])
@pytest.mark.parametrize("a", [.03,.25,.8])
def test_population_formula_agrees_with_independent_full_matrix_likelihood(n, a):
    mixture = matched_lag_mixture(1/3,LENGTH)
    times = np.arange(n)/3
    true = mixture_covariance(times, [1.], mixture)
    assumed = a**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    direct = np.linalg.slogdet(assumed)[1] + np.trace(np.linalg.solve(assumed,true))
    assert population_objective(a, true[0,1], n) == pytest.approx(direct, rel=2e-14)


def test_population_derivative_and_unique_grid_minimum():
    mixture = matched_lag_mixture(1/3,LENGTH)
    r1 = float(mixture.correlation(1/3))
    for a in (.02,.15,.4,.75):
        step = 1e-6
        fd = (population_objective(a+step,r1)-population_objective(a-step,r1))/(2*step)
        assert population_derivative(a,r1) == pytest.approx(fd, rel=2e-8, abs=1e-9)
    result = minimize_scalar(lambda a:population_objective(a,r1),bounds=(1e-5,.99),
                             method='bounded',options={'xatol':1e-13})
    assert result.success and result.x == pytest.approx(r1, abs=1e-8)
    lengths = np.geomspace(1/60,1.5,33)
    scores = population_objective(np.exp(-(1/3)/lengths),r1)
    assert np.argmin(scores) == 19
    assert population_derivative(r1,r1) == 0.
    assert not scores.flags.writeable
    assert population_objective(0.,0.,n=9) == 9.


def test_mixture_station_layout_spd_and_owned_covariance():
    times = np.array([.1,.3,.8])
    sd = np.array([1.,2.])
    mixture = ExponentialMixture(.1,1.4,.6)
    cov = mixture_covariance(times,sd,mixture)
    lags = np.abs(times[:,None]-times[None,:])
    expected = .6*np.exp(-lags/.1)+.4*np.exp(-lags/1.4)
    np.testing.assert_allclose(cov[:3,:3],expected,rtol=0,atol=0)
    np.testing.assert_allclose(cov[3:,3:],4*expected,rtol=0,atol=0)
    np.testing.assert_array_equal(cov[:3,3:],np.zeros((3,3)))
    np.testing.assert_array_equal(cov.diagonal(),[1,1,1,4,4,4])
    assert np.linalg.eigvalsh(cov).min() > 0
    assert cov.dtype == np.dtype('float64') and not cov.flags.writeable
    times[:],sd[:] = 900,900
    np.testing.assert_array_equal(cov.diagonal(),[1,1,1,4,4,4])


def test_kl_and_whitening_match_closed_diagonal_formula():
    truth = np.diag([1.,4.,9.])
    assumed = np.diag([2.,3.,8.])
    result = covariance_diagnostics(truth,assumed)
    ratio = np.diag(truth)/np.diag(assumed)
    expected = .5*np.sum(ratio-1-np.log(ratio))
    assert result['kl_true_to_assumed'] == pytest.approx(expected, rel=1e-14)
    assert result['whitening_spectral'] == pytest.approx(max(abs(ratio-1)), rel=1e-14)
    assert result['whitening_frobenius'] == pytest.approx(np.linalg.norm(ratio-1), rel=1e-14)
    json.dumps(result,allow_nan=False)
    np.testing.assert_array_equal(truth,np.diag([1.,4.,9.]))


def test_kl_general_matrix_congruence_and_near_identity_stability():
    truth = np.array([[2.,.5],[.5,1.]])
    assumed = np.array([[1.,-.1],[-.1,3.]])
    result = covariance_diagnostics(truth,assumed)
    direct = .5*(np.trace(np.linalg.solve(assumed,truth))-2
                  + np.linalg.slogdet(assumed)[1]-np.linalg.slogdet(truth)[1])
    assert result['kl_true_to_assumed'] == pytest.approx(direct,rel=1e-14)
    transform = np.array([[2.,.3],[-.1,.8]])
    transformed = covariance_diagnostics(transform@truth@transform.T,
                                         transform@assumed@transform.T)
    assert transformed['kl_true_to_assumed'] == pytest.approx(direct,rel=2e-14)
    same = covariance_diagnostics(truth,truth)
    assert 0 <= same['kl_true_to_assumed'] < 1e-26
    near = covariance_diagnostics(np.eye(2)*(1+1e-7),np.eye(2))
    assert near['kl_true_to_assumed'] == pytest.approx(1e-14/2,rel=1e-6)


@pytest.mark.parametrize("matrix", [np.array([[1,2],[2,1]]), np.zeros((2,2)),
    np.diag([1.,1e-12]), np.array([[1,.1],[0,1]]), [[1.,np.nan],[np.nan,1]],
    [[1.,np.inf],[np.inf,1]], [[True,False],[False,True]],
    np.eye(2,dtype=complex), np.array([[1,0],[0,1]],dtype=object), [[1,2,3]], []])
def test_real_guards_reject_invalid_spd_symmetry_condition_or_dtype(matrix):
    with pytest.raises(ValueError):
        covariance_diagnostics(matrix,np.eye(2))


def test_real_guards_prevent_ill_conditioned_mixture_without_jitter():
    with pytest.raises(ValueError):
        mixture_covariance([0,1e-12,2e-12],[1],ExponentialMixture(1,2,.5))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.eye(2),np.eye(3))
    with pytest.raises(TypeError):
        covariance_diagnostics(np.eye(2),np.eye(2),guards={})


@pytest.mark.parametrize("parameters", [dict(fast_hours=0),dict(fast_hours=np.nan),
    dict(slow_hours=np.inf),dict(fast_hours=2,slow_hours=1),dict(fast_hours=1,slow_hours=1),
    dict(fast_weight=0),dict(fast_weight=1),dict(fast_weight=-.1),dict(fast_weight=True),
    dict(fast_hours='0.1'),dict(fast_weight=1+0j)])
def test_invalid_mixture_parameters(parameters):
    with pytest.raises(ValueError):
        ExponentialMixture(**parameters)


@pytest.mark.parametrize("parameters", [dict(delta_hours=0,population_length_hours=.2),
    dict(delta_hours=True,population_length_hours=.2),dict(delta_hours=1/3,population_length_hours=1),
    dict(delta_hours=1/3,population_length_hours=1/12),dict(delta_hours=1e300,population_length_hours=.2),
    dict(delta_hours=1e-300,population_length_hours=.2)])
def test_unresolved_or_degenerate_lag_match_rejected(parameters):
    with pytest.raises(ValueError):
        matched_lag_mixture(**parameters)


@pytest.mark.parametrize("times,sd", [([0,0,1],[1]),([1,0],[1]),([0],[1]),
    ([[0,1]],[1]),([0,np.nan],[1]),([0,np.inf],[1]),([0,1],[]),
    ([0,1],[0]),([0,1],[-1]),([0,1],[np.inf]),([0,1],[[1]]),
    ([0,1],[1e-300]),([0,1],[1e300]),([0,1],[True])])
def test_invalid_covariance_time_or_station_layout(times,sd):
    with pytest.raises(ValueError):
        mixture_covariance(times,sd,ExponentialMixture())


@pytest.mark.parametrize("a,lag1,n", [(-.1,.2,9),(1,.2,9),(np.nan,.2,9),(.2,-.1,9),
    (.2,1,9),(.2,.2,1),(.2,.2,True),(.2,.2,9.),(True,.2,9),('0.2',.2,9)])
def test_invalid_population_inputs(a,lag1,n):
    for method in (population_objective,population_derivative):
        with pytest.raises(ValueError):
            method(a,lag1,n)


@pytest.mark.parametrize("parameters", [dict(cond2_max=.5),dict(cond2_max=np.inf),
    dict(relative_symmetry=0),dict(whitening_spectral=1),dict(cholesky_backward=True)])
def test_invalid_guard_configuration(parameters):
    with pytest.raises(ValueError):
        CovarianceGuards(**parameters)
