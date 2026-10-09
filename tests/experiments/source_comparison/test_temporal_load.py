"""Проверки средних нагрузок P1 и производных полулинейного прогноза."""
import numpy as np
import pytest
from scipy.integrate import quad

from adrkit.backends.cached import CachedSolver
from adrkit.loads import IntervalAverageLoad, interval_hat_integrals
from adrkit.predictions import GridPrediction, grid_observation, solver_state_space
from adrkit.inverse.regularization import p1_matrices
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace


def load_for(basis, *, steps=672, history=None, initial=0., scale=2.3):
    return IntervalAverageLoad(basis, np.linspace(0, 3.5, steps+1), origin=-.5,
        q_reference=scale, history_integral=history or (lambda a, b: 0.),
        history_initial=initial, source_unit="source_unit")


def test_nonuniform_integrals_against_independent_scalar_quadrature():
    knots = np.array([0., .23, .81, 1.7])
    edges = np.array([-.5, -.1, .07, .64, 1.2, 1.9])
    actual = interval_hat_integrals(P1Basis(knots, time_unit="h"), edges)
    expected = np.zeros_like(actual)
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        for j in range(len(knots)):
            def hat(t):
                if j > 0 and knots[j-1] <= t <= knots[j]:
                    return (t-knots[j-1])/(knots[j]-knots[j-1])
                if j+1 < len(knots) and knots[j] <= t <= knots[j+1]:
                    return (knots[j+1]-t)/(knots[j+1]-knots[j])
                return 0.
            expected[i, j] = quad(hat, left, right,
                points=[k for k in knots if left < k < right], epsabs=1e-13)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-16)


@pytest.mark.parametrize("nodes,ratio", [(73, 8), (145, 4), (289, 2)])
def test_full_rank_endpoint_mass_and_first_rows(nodes, ratio):
    basis = P1Basis(np.linspace(0, 3, nodes), time_unit="h")
    load = load_for(basis)
    U = load.averages
    assert np.linalg.matrix_rank(U) == nodes  # Ранг отображения нагрузки, до пространственных наблюдений.
                                              # Якобиан наблюдений здесь не проверяется.
    np.testing.assert_array_equal(U[:97], 0.)
    np.testing.assert_allclose(U[97, :2], [1-1/(2*ratio), 1/(2*ratio)], atol=2e-14)
    np.testing.assert_allclose(U[98, :2], [1-3/(2*ratio), 3/(2*ratio)], atol=2e-14)
    M, _ = p1_matrices(basis)
    np.testing.assert_allclose(U[1:].sum(axis=0)/192, M @ np.ones(nodes),
                               rtol=2e-14, atol=2e-16)
    np.testing.assert_allclose(U[97:].sum(axis=1), 1., rtol=0, atol=9e-14)
    # Интерполяция по концам изменила бы нагрузки, в том числе первую базисную функцию.
    assert not np.allclose(U[97], basis.values([1/192])[0])


@pytest.mark.parametrize("coarse_nodes,fine_nodes", [(73, 145), (145, 289)])
def test_prolongation_load_and_full_h1_galerkin(coarse_nodes, fine_nodes):
    coarse = P1Basis(np.linspace(0, 3, coarse_nodes), time_unit="h")
    fine = P1Basis(np.linspace(0, 3, fine_nodes), time_unit="h")
    P = coarse.values(fine.knots)
    np.testing.assert_allclose(load_for(fine).averages @ P, load_for(coarse).averages,
                               rtol=0, atol=3e-14)
    Mc, Kc = p1_matrices(coarse)
    Mf, Kf = p1_matrices(fine)
    for coarse_gram, fine_gram in [(Mc, Mf), (Kc, Kf), (Mc+.25**2*Kc, Mf+.25**2*Kf)]:
        np.testing.assert_allclose(P.T @ fine_gram @ P, coarse_gram, rtol=2e-14, atol=2e-14)
    assert np.ones(coarse_nodes) @ (Mc+.25**2*Kc) @ np.ones(coarse_nodes) > 2.99


def test_history_clipping_and_crossing_step_without_endpoint_constraint():
    basis = P1Basis([0., 1., 3.], time_unit="h")
    # Предыстория задана экспонентой.
    # Её интеграл вычисляется аналитически.
    history = lambda left, right: np.exp(right)-np.exp(left)
    times = np.array([0., .3, .7, 3.5])  # Физический интервал [-.2,.2] пересекает ноль.
    load = IntervalAverageLoad(basis, times, origin=-.5, q_reference=2.,
        history_integral=history, history_initial=np.exp(-.5), source_unit="source_unit")
    expected = [np.exp(-.5), (np.exp(-.2)-np.exp(-.5))/.3,
                (1-np.exp(-.2))/.4, 0.]
    np.testing.assert_allclose(load.history, expected, rtol=2e-15)
    np.testing.assert_allclose(load.averages[2], [.18/.4, .02/.4, 0.], atol=2e-16)
    np.testing.assert_array_equal(load.averages[0], 0.)
    # Первый коэффициент профиля можно менять независимо от предыстории.
    assert load.predict([7., 1., 2.])[0] == np.exp(-.5)


def test_affine_derivatives_transpose_and_no_extra_dt_or_mass():
    load = load_for(P1Basis([0., .2, 1., 3.], time_unit="h"))
    rng = np.random.default_rng(512)
    a, h, z = rng.normal(size=4), rng.normal(size=4), rng.normal(size=673)
    np.testing.assert_allclose(load.predict(a+h)-load.predict(a), load.jvp(a, h), atol=3e-15)
    np.testing.assert_allclose(z @ load.jvp(a, h), h @ load.vjp(a, z), rtol=2e-14)
    U = load.averages
    U[:] = 99  # Возвращённый диагностический массив не должен изменять операцию.
    assert not np.any(load.averages == 99)


def nonlinear_fixture(nodes=5):
    solver = CachedSolver(dict(horizon=3.5, diffusion=.12, velocity=[.1, -.07],
        linear_loss=.08, reaction_gamma=.2, reaction_c_star=.9),
        interior_points=3, time_steps=672, residual_tolerance=1e-13)
    basis = P1Basis(np.linspace(0, 3, nodes), time_unit="h")
    load = load_for(basis, history=lambda a, b: .4*(b-a), initial=.4)
    S = np.array([[.1, .2, .1, .2, .6, .2, .1, .2, .1],
                  [.2, .1, .3, .1, .2, .4, .1, .3, .2]])
    W = np.zeros((3, 673))
    W[np.arange(3), [96, 288, 672]] = 1.
    obs = ArraySpace((6,), axes=("observation",),
        coordinates=(("A0", "A1", "A3", "B0", "B1", "B3"),), units="state_unit")
    state_space = solver_state_space(solver, state_unit="state_unit")
    observation = grid_observation(S, W, state_space=state_space, codomain=obs)
    prediction = GridPrediction(solver, load, observation).capabilities()
    return solver, load, S, W, prediction


def test_actual_semilinear_chain_manual_controls_taylor_and_transpose():
    solver, load, S, W, prediction = nonlinear_fixture()
    a = np.array([.7, .3, 1.2, .9, .4])
    h = np.array([.2, -.1, .3, -.4, .1])
    z = np.array([.2, -.5, .3, .7, -.8, .1])
    # Эталонные средние нагрузки вычисляются отдельной квадратурой.
    # Она разбивается по узлам P1 до решения прямой задачи.
    clock = solver.times-.5
    knots = np.linspace(0, 3, len(a))
    controls = np.zeros(673)
    controls[:97] = .4
    for k in range(97, 673):
        left, right = clock[k-1:k+1]
        cuts = np.r_[left, knots[(knots > left) & (knots < right)], right]
        controls[k] = 2.3*np.sum(np.diff(cuts)*(
            np.interp(cuts[:-1], knots, a)+np.interp(cuts[1:], knots, a))/2)/solver.dt
    states = solver.solve_controls(controls).states
    expected = np.r_[W @ (states @ S[0]), W @ (states @ S[1])]
    actual = prediction.predict.predict(a)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    derivative = prediction.jvp.jvp(a, h)
    epsilon = 1e-4
    fd = (prediction.predict.predict(a+epsilon*h)-prediction.predict.predict(a-epsilon*h))/(2*epsilon)
    np.testing.assert_allclose(derivative, fd, rtol=2e-7, atol=2e-10)
    errors = [np.linalg.norm(prediction.predict.predict(a+e*h)-actual-e*derivative)
              for e in (.04, .02, .01)]
    assert errors[0]/errors[1] > 3.8 and errors[1]/errors[2] > 3.8
    np.testing.assert_allclose(z @ derivative, h @ prediction.vjp.vjp(a, z), rtol=2e-10, atol=2e-12)
    assert not np.allclose(2*derivative, fd, rtol=2e-7, atol=2e-10)


def test_nonlinear_forward_is_unchanged_by_p1_prolongation():
    _, _, _, _, coarse = nonlinear_fixture(73)
    _, _, _, _, fine = nonlinear_fixture(145)
    a = .5 + .2*np.cos(np.linspace(0, 3, 73))
    P = P1Basis(np.linspace(0, 3, 73), time_unit="h").values(np.linspace(0, 3, 145))
    np.testing.assert_allclose(coarse.predict.predict(a), fine.predict.predict(P @ a), rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("edges", [[0., 0., 1.], [0., np.nan], [0., 1j]])
def test_invalid_clock_rejected(edges):
    with pytest.raises(ValueError):
        interval_hat_integrals(P1Basis([0., 1.], time_unit="h"), edges)
