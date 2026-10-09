"""Проверки нормировки ядер, производных ширины и порядка наблюдений."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.integrate import quad

from adrkit.observations.grid import GridObservation
from adrkit.spaces import ArraySpace
from experiments.observation_sensitivity import CompactKernel, GaussianKernel, spatial_weights


@pytest.mark.parametrize("family", [GaussianKernel, CompactKernel])
@pytest.mark.parametrize("width", [0.5, 1.0, 2.0])
def test_independent_radial_integrals_and_coordinate_moments(family, width):
    kernel = family(width)
    limit = np.inf if kernel.support_radius is None else kernel.support_radius
    value = lambda r: kernel.density([[r, 0.0]])[0]
    mass = quad(lambda r: 2*np.pi*r*value(r), 0, limit, epsabs=1e-11, epsrel=1e-11)[0]
    # Угловой интеграл cos² равен π: получается второй момент одной координаты.
    variance = quad(lambda r: np.pi*r**3*value(r), 0, limit, epsabs=1e-11, epsrel=1e-11)[0]
    derivative_mass = quad(lambda r: 2*np.pi*r*kernel.log_width_derivative([[r, 0]])[0],
                           0, limit, epsabs=1e-10, epsrel=1e-10)[0]
    assert mass == pytest.approx(1.0, rel=1e-10, abs=1e-12)
    assert variance == pytest.approx(width**2, rel=1e-10, abs=1e-12)
    assert kernel.coordinate_variance == width**2
    assert abs(derivative_mass) < 1e-10


@pytest.mark.parametrize("family", [GaussianKernel, CompactKernel])
def test_analytic_log_width_derivative_at_fixed_translated_points(family):
    center = np.array([1.7, -2.3])
    points = center + np.array([[0,0], [.2,.4], [1.3,-.8], [2.,0], [2.7,.05], [5.,-2.]])
    kernel = family(1.0)
    analytic = kernel.log_width_derivative(points, center)
    errors = []
    for h in (1e-3, 5e-4, 2.5e-4):
        fd = (family(np.exp(h)).density(points, center)
              - family(np.exp(-h)).density(points, center))/(2*h)
        errors.append(np.linalg.norm(fd-analytic, np.inf))
    assert errors[-1] < errors[0]/12
    assert errors[-1]/np.linalg.norm(analytic, np.inf) < 1e-5
    # Вблизи плоской границы компактного носителя малая абсолютная производная имеет большую
    # относительную ошибку усечения на крупных шагах. Она проверяется независимо.
    h = 1e-6
    fd = (family(np.exp(h)).density(points, center)
          - family(np.exp(-h)).density(points, center))/(2*h)
    np.testing.assert_allclose(fd, analytic, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(kernel.density(points, center), kernel.density(points-center), rtol=0, atol=0)


def test_compact_boundary_is_flat_zero_and_density_nonnegative():
    kernel = CompactKernel(1.)
    a = kernel.support_radius
    points = [[0,0], [a*.98,0], [a*(1-1e-6),0], [a,0], [a*1.01,0]]
    density = kernel.density(points)
    derivative = kernel.log_width_derivative(points)
    assert density[0] > density[1] >= 0
    np.testing.assert_array_equal(density[2:], [0.,0.,0.])
    np.testing.assert_array_equal(derivative[2:], [0.,0.,0.])


def test_gaussian_raw_rows_match_frozen_builder_bitwise_and_are_not_normalized():
    points = np.array([[-4.75,-2.], [.1,.2], [0.,0.], [3.,-1.], [-11.,-2.]])
    centers = np.array([[0,0], [-4.807498961147989,-2.0443267416397575]])
    width, area = 1.0, .25**2
    expected = np.array([area*np.exp(-np.sum((points-center)**2,axis=1)/(2*width**2))
                         /(2*np.pi*width**2) for center in centers])
    weights = spatial_weights(GaussianKernel(width), points, centers, area)
    np.testing.assert_array_equal(weights, expected)
    assert not np.allclose(weights.sum(axis=1), 1.)
    assert weights.dtype == np.dtype('float64') and not weights.flags.writeable
    points[:] = centers[:] = 900
    np.testing.assert_array_equal(weights, expected)


@pytest.mark.parametrize("family", [GaussianKernel, CompactKernel])
def test_raw_row_scale_derivative_and_actual_receiver_major_adjoint(family):
    points = np.array([[0.,0.], [.5,.2], [-1.,.4]])
    centers = np.array([[.3,-.4], [1.,0.]])
    area = .0625
    weights = spatial_weights(family(1.), points, centers, area)
    derivative = spatial_weights(family(1.), points, centers, area, derivative=True)
    step = 1e-5
    fd = (spatial_weights(family(np.exp(step)), points, centers, area)
          - spatial_weights(family(np.exp(-step)), points, centers, area))/(2*step)
    np.testing.assert_allclose(fd, derivative, rtol=1e-7, atol=1e-12)
    temporal = np.array([[0.,1.,0.], [.2,.3,.5]])
    domain = ArraySpace((3,3), axes=('time','node'),
                        coordinates=((0.,.5,1.),tuple(map(tuple,points))), units='ug/m^3')
    output = ArraySpace((4,), axes=('observation',),
                        coordinates=(('A/snapshot','A/mean','B/snapshot','B/mean'),), units='ug/m^3')
    observation = GridObservation(weights, temporal, domain=domain, codomain=output)
    mapping = observation.capabilities()
    states = np.array([[.2,1.,3.], [2.,-.1,.5], [.4,1.2,-.3]])
    dual = np.array([.7,-.2,.3,.4])
    expected = np.array([sum(temporal[t,k]*weights[r,j]*states[k,j]
                             for k in range(3) for j in range(3))
                         for r in range(2) for t in range(2)])
    predicted = mapping.predict.predict(states)
    np.testing.assert_allclose(predicted, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(dual@predicted, np.sum(states*mapping.vjp.vjp(states,dual)),
                               rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("family", [GaussianKernel, CompactKernel])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, 1+0j, '1', [1.], 1e-300, 1e300])
def test_invalid_or_unrepresentable_width_fails_at_construction(family, value):
    with pytest.raises(ValueError):
        family(value)


@pytest.mark.parametrize("family", [GaussianKernel, CompactKernel])
def test_scalar_numeric_ownership_and_frozen_parameters(family):
    original = np.array(1., dtype=np.float32)
    kernel = family(original)
    original[...] = 3
    assert kernel.width == 1. and isinstance(kernel.width, float)
    with pytest.raises(FrozenInstanceError):
        kernel.width = 2


@pytest.mark.parametrize("points", [[], [1,2], [[0,0,0]], [[np.nan,0]], [[0,np.inf]],
                                      [[True,False]], [['0','0']], [[0j,0j]],
                                      np.array([[0,0]], dtype=object)])
def test_invalid_point_geometry_is_rejected(points):
    with pytest.raises(ValueError):
        CompactKernel(1).density(points)


@pytest.mark.parametrize("center", [[], [[0,0]], [0,0,0], [0,np.nan], [True,False]])
def test_invalid_center_is_rejected(center):
    with pytest.raises(ValueError):
        GaussianKernel(1).density([[0,0]], center)


@pytest.mark.parametrize("area", [0,-1,np.nan,np.inf,True,'1',[1]])
def test_invalid_cell_area_is_rejected(area):
    with pytest.raises(ValueError):
        spatial_weights(GaussianKernel(1), [[0,0]], [[0,0]], area)


def test_spatial_builder_rejects_malformed_layouts_and_options():
    for centers in ([], [0,0], [[0,0,0]], [[0,np.nan]]):
        with pytest.raises(ValueError):
            spatial_weights(GaussianKernel(1), [[0,0]], centers, 1.)
    with pytest.raises(TypeError):
        spatial_weights(object(), [[0,0]], [[0,0]], 1.)
    with pytest.raises(TypeError):
        spatial_weights(GaussianKernel(1), [[0,0]], [[0,0]], 1., derivative=1)


def test_finite_extreme_coordinates_do_not_create_nan_or_false_gaussian_zero():
    gaussian = GaussianKernel(5e153)
    value = gaussian.density([[1.5e154,0.]])[0]
    # Квадрат координаты переполняется, но radius/width=3, а плотность представима субнормальным
    # float; возвращать ноль без проверки нельзя.
    expected = np.exp(-4.5)/(2*np.pi*(5e153)**2)
    assert expected > 0
    assert value == pytest.approx(expected, rel=1e-12, abs=0)
    for kernel in (GaussianKernel(1), CompactKernel(1)):
        np.testing.assert_array_equal(kernel.density([[1e308,0]],[-1e308,0]), [0])
        np.testing.assert_array_equal(kernel.log_width_derivative([[1e308,0]],[-1e308,0]), [0])


@pytest.mark.parametrize("family", [GaussianKernel,CompactKernel])
def test_representable_scaled_tails_are_not_lost_to_intermediate_underflow(family):
    kernel = family(1.)
    if family is GaussianKernel:
        point = np.array([[np.sqrt(1600.),0.]])
        exponent, normalizer = -800., 2*np.pi
        multiplier = 1598.
    else:
        point = np.array([[kernel.support_radius*np.sqrt(1-1/800),0.]])
        u = (point[0,0]/kernel.support_radius)**2
        exponent = -1/(1-u)
        # Независимая радиальная квадратура для нормировки непрерывного ядра.
        i0 = quad(lambda x: np.exp(-1/(1-x)),0,1,epsabs=1e-13)[0]
        normalizer = np.pi*kernel.support_radius**2*i0
        multiplier = -2+2*u/(1-u)**2
    expected = np.exp(np.log(1e308)+exponent-np.log(normalizer))
    assert expected > 0
    assert spatial_weights(kernel,point,[[0,0]],1e308)[0,0] == pytest.approx(expected,rel=1e-12)
    assert spatial_weights(kernel,point,[[0,0]],1e308,derivative=True)[0,0] == pytest.approx(
        expected*multiplier,rel=1e-12)
