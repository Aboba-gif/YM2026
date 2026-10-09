"""Проверки моментов и крайних масштабов арифметики E06.

Используются масштабные тождества, ковариации вне семейства смеси и сравнения
с Decimal при 80 значащих цифрах."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
from scipy.integrate import quad

from experiments.observation_sensitivity import CompactKernel, GaussianKernel, covariance_diagnostics, population_derivative, population_objective, spatial_weights


def test_exact_zero_scale_derivative_survives_overflowing_intermediate_weight():
    # Здесь r²/s² равно 2 в float64. Нормированная плотность конечна, её производная по
    # масштабу точно равна нулю. Большой конечный множитель квадратуры не должен превращать
    # представимый ноль в NaN через overflow * zero.
    width = 1e-154
    points = np.array([[width, width]])
    ratio = np.sum(points**2, axis=1) / width**2
    np.testing.assert_array_equal(ratio, [2.])
    kernel = GaussianKernel(width)
    np.testing.assert_array_equal(kernel.log_width_derivative(points), [0.])
    actual = spatial_weights(kernel, points, [[0., 0.]], 1e308, derivative=True)
    np.testing.assert_array_equal(actual, [[0.]])


@pytest.mark.parametrize('scale', [1e160, 1e200, 1e300])
def test_frobenius_diagnostic_is_finite_when_its_mathematical_value_is_finite(scale):
    # Оба числа обусловленности SPD равны единице. Норма Фробениуса sqrt(2)*(s-1) представима, хотя
    # прямое возведение её элементов в квадрат переполняется.
    truth, assumed = scale*np.eye(2), np.eye(2)
    with np.errstate(over='ignore'):
        result = covariance_diagnostics(truth, assumed)
    assert result['kl_true_to_assumed'] == pytest.approx(scale-1-math.log(scale), rel=3e-15)
    assert result['whitening_spectral'] == pytest.approx(scale-1, rel=3e-15)
    assert result['whitening_frobenius'] == pytest.approx(math.hypot(scale-1, scale-1), rel=3e-15)
    assert all(math.isfinite(result[k]) for k in ('kl_true_to_assumed',
                                                 'whitening_spectral',
                                                 'whitening_frobenius'))


@pytest.mark.parametrize('family', [GaussianKernel, CompactKernel])
@pytest.mark.parametrize('width', [.5, 2.])
def test_second_and_fourth_moments_have_expected_log_width_derivatives(family, width):
    # Для этих масштабных ядер конечные моменты M_k(s)=s^k M_k(1), поэтому
    # dM_k/dlog(s)=k M_k. Проверяются знак и чувствительность к масштабу, а не только нулевая
    # производная массы.
    kernel = family(width)
    limit = kernel.support_radius or np.inf
    density = lambda r: kernel.density([[r, 0.]])[0]
    derivative = lambda r: kernel.log_width_derivative([[r, 0.]])[0]
    radial_second = quad(lambda r: 2*np.pi*r**3*density(r), 0., limit,
                         epsabs=1e-10, epsrel=1e-11)[0]
    second_derivative = quad(lambda r: 2*np.pi*r**3*derivative(r), 0., limit,
                             epsabs=1e-10, epsrel=1e-11)[0]
    fourth = quad(lambda r: 2*np.pi*r**5*density(r), 0., limit,
                  epsabs=1e-10, epsrel=1e-11)[0]
    fourth_derivative = quad(lambda r: 2*np.pi*r**5*derivative(r), 0., limit,
                             epsabs=1e-10, epsrel=1e-11)[0]
    assert radial_second == pytest.approx(2*width**2, rel=2e-10)
    assert second_derivative == pytest.approx(2*radial_second, rel=2e-10)
    assert fourth_derivative == pytest.approx(4*fourth, rel=2e-10)


@pytest.mark.parametrize('family', [GaussianKernel, CompactKernel])
@pytest.mark.parametrize('scale', [1e-100, 1e-20, 1e20, 1e100])
def test_dimensionless_raw_weights_are_scale_equivariant(family, scale):
    points = np.array([[.1, .2], [.75, -.5], [2.3, .2]])
    centers = np.array([[0., 0.], [.2, -.4]])
    for derivative in [False, True]:
        unit = spatial_weights(family(1.), points, centers, 1., derivative=derivative)
        scaled = spatial_weights(family(scale), points*scale, centers*scale,
                                 scale**2, derivative=derivative)
        np.testing.assert_allclose(scaled, unit, rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize('a', [0., .3, .97, np.nextafter(1., 0.)])
@pytest.mark.parametrize('lag1', [0., .2, np.nextafter(1., 0.)])
def test_population_extremes_against_80_digit_decimal(a, lag1):
    with localcontext() as ctx:
        ctx.prec = 80
        x, r, n = Decimal.from_float(a), Decimal.from_float(lag1), Decimal(9)
        denominator = 1-x*x
        objective = (n-1)*denominator.ln() + (n+(n-2)*x*x-2*(n-1)*x*r)/denominator
        derivative = 2*(n-1)*(x-r)*(1+x*x)/(denominator*denominator)
    assert population_objective(a, lag1) == pytest.approx(float(objective), rel=2e-14, abs=1e-13)
    assert population_derivative(a, lag1) == pytest.approx(float(derivative), rel=2e-14, abs=1e-13)


@pytest.mark.parametrize('a', [.1, .5, .9])
def test_population_trace_ignores_higher_lags_for_nonmixture_truth(a):
    # Две строго диагонально преобладающие ковариационные матрицы Тёплица, не из
    # двухэкспоненциального генератора, имеют одинаковые диагональ и первый лаг. Их различные
    # дальние лаги не входят в след с матрицей точности AR(1).
    n, r = 7, .2
    lag = np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    truth_a = np.eye(n)+r*(lag == 1)
    truth_b = truth_a+.08*(lag == 2)-.03*(lag == 3)
    assumed = a**lag
    direct = []
    for truth in [truth_a, truth_b]:
        assert np.linalg.eigvalsh(truth).min() > 0
        direct.append(np.linalg.slogdet(assumed)[1] + np.trace(np.linalg.solve(assumed, truth)))
    np.testing.assert_allclose(direct, population_objective(a, r, n), rtol=2e-14, atol=1e-13)


def _decimal_kl_2_by_2(truth, assumed):
    # Явные обратные матрицы и определители 2×2 не используют разложения Холецкого и SVD.
    with localcontext() as ctx:
        ctx.prec = 80
        t = [[Decimal.from_float(float(v)) for v in row] for row in truth]
        a = [[Decimal.from_float(float(v)) for v in row] for row in assumed]
        dt = t[0][0]*t[1][1]-t[0][1]*t[1][0]
        da = a[0][0]*a[1][1]-a[0][1]*a[1][0]
        trace = (a[1][1]*t[0][0]+a[0][0]*t[1][1]
                 -a[0][1]*t[1][0]-a[1][0]*t[0][1])/da
        return float((trace-2+da.ln()-dt.ln())/2)


def test_kl_orientation_against_decimal_for_noncommuting_matrices():
    truth = np.array([[5., 2.], [2., 3.]])
    assumed = np.array([[2., .3], [.3, 1.]])
    assert not np.allclose(truth@assumed, assumed@truth)
    forward = covariance_diagnostics(truth, assumed)['kl_true_to_assumed']
    reverse = covariance_diagnostics(assumed, truth)['kl_true_to_assumed']
    assert forward == pytest.approx(_decimal_kl_2_by_2(truth, assumed), rel=2e-14)
    assert reverse == pytest.approx(_decimal_kl_2_by_2(assumed, truth), rel=2e-14)
    assert abs(forward-reverse) > .1


def test_tiny_negative_eigenvalue_is_rejected_without_jitter():
    with pytest.raises(ValueError):
        covariance_diagnostics(np.diag([1., -1e-20]), np.eye(2))
