"""Сборка прямой задачи, синтетических наблюдений и метрик ошибки источника."""
from types import SimpleNamespace

import numpy as np

from adrkit.backends.cached import CachedSolver
from adrkit.config.validation import JSONRecord
from experiments.source_comparison.observations import build_observations
from experiments.source_comparison.truth import interval_controls
from adrkit.inverse.regularization import p1_matrices
from adrkit.observations.grid import apply_observation
from adrkit.loads import (
    IntervalAverageLoad, interval_hat_integrals, DisclosedHistory, disclose_history,
)
from adrkit.predictions import (
    GridPrediction, grid_observation, solver_state_space,
)
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace
from .sources import first_moment, unknown_L2_squared


def make_solver(spec, grid_name, gamma):
    """Создать решатель состояния для заданной сетки и реакции.

    Parameters
    ----------
    spec : dict
        Коэффициенты модели, пространственные сетки в км и горизонт в
        часах.
    grid_name : str
        Ключ сетки в разделе grids.
    gamma : float
        Коэффициент нелинейной реакции, в обратных часах.

    Returns
    -------
    solver : CachedSolver
        Решатель с временными шагами и допуском прямой невязки из
        конфигурации.
    """

    mesh = spec["grids"][grid_name]
    bounds = np.asarray(mesh["bounds_km"], dtype=float)
    counts = np.diff(bounds.reshape(2,2), axis=1).ravel()/mesh["spacing_km"]
    if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-11):
        raise ValueError("Domain must align with grid")
    shape = np.rint(counts).astype(int)-1
    model = {k:spec["model"][k] for k in ("diffusion", "velocity", "linear_loss",
        "reaction_c_star", "source_position", "background", "horizon")}
    model["reaction_gamma"] = gamma
    return CachedSolver(model, interior_points=shape, time_steps=mesh["steps"],
                        domain=bounds, residual_tolerance=spec["solver"]["forward_tolerance"])


def make_observations(spec, solver, condition):
    """Построить веса пространственных и временных наблюдений концентрации.

    Parameters
    ----------
    spec : dict
        Параметры модели и расположение постов.
    solver : CachedSolver
        Решатель с пространственной и временной сетками состояния.
    condition : dict
        Условия опыта: сдвиг поста КрАЗ и временной оператор.
        ``snapshot`` выбирает момент, ``average20`` усредняет состояние
        за предшествующие 20 минут.

    Returns
    -------
    spatial : ndarray, shape (4, solver.size)
        Безразмерные пространственные веса постов с площадью ячейки.
    temporal : ndarray, shape (72, solver.nt + 1)
        Безразмерные временные веса наблюдений.
    """
    from .config import DENSE_TIMES
    centers = np.array(spec["observations"]["centers_km"], dtype=float, copy=True)
    relocation = float(condition["relocation_km"])
    if relocation:
        velocity = np.asarray(spec["model"]["velocity"])
        speed = np.linalg.norm(velocity)
        if speed == 0:
            raise ValueError("Along-wind relocation undefined for zero velocity")
        centers[3] += relocation*velocity/speed
    metadata = dict(spec["observations"], centers_km=centers.tolist(),
        full_times_hours=DENSE_TIMES.tolist(), regimes=[dict(id="dense",
            time_indices_zero_based=list(range(72)), rows=288)])
    # Локальное описание геометрии передаётся общему построителю наблюдений.
    view = SimpleNamespace(document=JSONRecord(dict(execution_config=dict(
        model=dict(extended_start_hours=spec["model"]["origin_hours"]), observations=metadata))))
    design = build_observations(view, solver, regime="dense")
    temporal = design.time_weights
    if condition["temporal_H"] == "average20":
        state_basis = P1Basis(solver.times+spec["model"]["origin_hours"], time_unit="h")
        temporal = np.vstack([3*interval_hat_integrals(state_basis, [t-1/3,t])[0]
                              for t in DENSE_TIMES])
        if not np.allclose(temporal.sum(axis=1), 1., rtol=0, atol=2e-12):
            raise ValueError("Temporal averaging interval not covered by state clock")
    elif condition["temporal_H"] != "snapshot":
        raise ValueError("Unknown temporal observation operator")
    return design.space_weights, temporal


class Prediction(GridPrediction):
    """Прогноз концентрации для профиля P1 на интервале 0–3 ч.

    Parameters
    ----------
    spec : dict
        Параметры модели и масштаб интенсивности источника.
    condition : dict
        Число узлов профиля и пространственно-временной оператор наблюдений.
    solver : CachedSolver
        Решатель состояния на сетке обратной задачи.
    history : DisclosedHistory
        Известная предыстория источника до нулевого времени.

    Attributes
    ----------
    basis : P1Basis
        Базис безразмерных узловых значений на интервале от 0 до 3 часов.
    space_weights, time_weights : ndarray
        Пространственные и временные веса наблюдений.

    Notes
    -----
    Решатель и нагрузка копируются базовым GridPrediction. Свойство trajectory
    возвращает только неизменяемый StateCertificate.
    """

    def __init__(self, spec, condition, solver, history):
        self.basis = P1Basis(np.linspace(0, 3, condition["nodes"]), time_unit="h")
        load = IntervalAverageLoad(self.basis, solver.times,
            origin=spec["model"]["origin_hours"], q_reference=spec["Qref"],
            history_integral=history.integral, history_initial=history.initial,
            source_unit="C*km^2/hour")
        spatial, temporal = make_observations(spec, solver, condition)
        state_space = solver_state_space(solver, state_unit="ug/m^3")
        row_count = spatial.shape[0]*temporal.shape[0]
        result_space = ArraySpace((row_count,), axes=("observation",),
                                 coordinates=(np.arange(row_count),),
                                 units=state_space.units, dtype=state_space.dtype)
        observation = grid_observation(spatial, temporal,
                                       state_space=state_space, codomain=result_space)
        super().__init__(solver, load, observation)
        self.space_weights, self.time_weights = spatial, temporal


def source_scores(source, basis, coefficients, qref):
    r"""Сравнить восстановленный профиль P1 с аналитическим источником.

    Parameters
    ----------
    source : FiniteRelease, BiExponential, TwoPulse or ScaledSource
        Заданный профиль с ненулевой нормой L².
    basis : P1Basis
        Кусочно-линейный базис на интервале от 0 до 3 часов.
    coefficients : array_like, shape (basis.size,)
        Безразмерные узловые значения восстановленного профиля.
    qref : float
        Положительный масштаб интенсивности в C·км²/ч; C — единица
        концентрации.

    Returns
    -------
    scores : dict of str to float
        Безразмерные ошибки E_q, relative_L2, signed_mass_error,
        absolute_mass_error и zero_prior_E_q. Интегралы estimated_mass и
        true_mass заданы в C·км².

    Notes
    -----
    Пусть :math:`q` — заданный профиль, :math:`\widehat q` — профиль P1
    с узловыми значениями :math:`Q_{\rm ref}a`, :math:`T=3` часа.
    Нормы и интегралы берутся по интервалу :math:`[0,T]`:

    .. math::

        E_q = \frac{\|\widehat q-q\|_{L^2}}{Q_{\rm ref}\sqrt{T}},
        \qquad E_{\rm rel} = \frac{\|\widehat q-q\|_{L^2}}{\|q\|_{L^2}},
        \qquad E_{\rm mass} =
        \frac{\int_0^T(\widehat q-q)\,dt}{Q_{\rm ref}T}.

    ``absolute_mass_error`` — модуль знаковой ошибки интеграла,
    ``zero_prior_E_q`` — E_q для нулевого восстановленного профиля.
    """
    coefficients = np.asarray(coefficients)*qref
    mass, _ = p1_matrices(basis)
    cross = np.zeros(basis.size)
    for i,(a,b) in enumerate(zip(basis.knots[:-1],basis.knots[1:])):
        integral, moment = source.integral(float(a),float(b)), first_moment(source,float(a),float(b))
        cross[i] += (b*integral-moment)/(b-a)
        cross[i+1] += (moment-a*integral)/(b-a)
    true_norm2 = unknown_L2_squared(source)
    difference2 = float(coefficients@mass@coefficients-2*coefficients@cross+true_norm2)
    tolerance = 1e-10*max(1.,true_norm2,float(coefficients@mass@coefficients))
    if difference2 < -tolerance:
        raise ValueError("Negative squared source error beyond roundoff")
    difference2 = max(difference2,0.)
    recovered_mass = float(np.trapezoid(coefficients,basis.knots))
    # Стандартный float нужен для последующей строгой записи метрик в JSON.
    truth_mass = float(source.integral(0.,3.))
    return dict(E_q=float(np.sqrt(difference2)/(qref*np.sqrt(3))),
        relative_L2=float(np.sqrt(difference2/true_norm2)),
        estimated_mass=recovered_mass, true_mass=truth_mass,
        signed_mass_error=(recovered_mass-truth_mass)/(qref*3),
        absolute_mass_error=abs(recovered_mass-truth_mass)/(qref*3),
        zero_prior_E_q=float(np.sqrt(true_norm2)/(qref*np.sqrt(3))))


class ProductionBackend:
    """Синтетические наблюдения и прогнозы для восстановления источника.

    Parameters
    ----------
    spec : dict
        Конфигурация моделей, сеток и наблюдений.
    source : FiniteRelease, BiExponential, TwoPulse or ScaledSource
        Аналитический профиль для синтетических наблюдений и оценки ошибки.
    """
    def __init__(self, spec, source):
        self.spec, self.source = spec, source
        self.truth_cache = {}

    def truth(self, condition):
        """Вычислить наблюдения заданного источника без добавленного шума.

        Результат повторно используется для тех же сетки генератора, реакции и
        геометрии наблюдений.

        Parameters
        ----------
        condition : dict
            Сетка генератора, реакция, временной оператор и сдвиг поста в км.

        Returns
        -------
        values : ndarray, shape (288,)
            Синтетические концентрации в мкг/м³; порядок: пост, затем время.
        residual : float
            Максимальная масштабированная невязка прямого решения.
        """

        grid=condition.get("truth_grid",self.spec["truth_grid"])
        key = (grid,condition["truth_gamma"],condition["temporal_H"],condition["relocation_km"])
        if key not in self.truth_cache:
            solver = make_solver(self.spec,grid,condition["truth_gamma"])
            weights, temporal = make_observations(self.spec,solver,condition)
            trajectory = solver.solve_controls(interval_controls(self.source,solver.times,
                origin=self.spec["model"]["origin_hours"]))
            self.truth_cache[key] = (apply_observation(trajectory.states,weights,temporal),
                                    float(trajectory.max_scaled_residual))
            # Кэш хранит только наблюдения и максимальную невязку, без траектории и LU.
        return self.truth_cache[key]

    def prediction(self, condition):
        """Создать прогноз для обратной задачи данного условия.

        Parameters
        ----------
        condition : dict
            Обратная сетка и реакция, число узлов профиля и оператор
            наблюдений.

        Returns
        -------
        prediction : Prediction
            Прогноз с известной предысторией; восстанавливаемые коэффициенты
            безразмерны.
        """

        solver = make_solver(self.spec,condition["grid"],condition["inverse_gamma"])
        history = disclose_history(self.source,solver.times,self.spec["model"]["origin_hours"])
        return Prediction(self.spec,condition,solver,history)

    def score_source(self, condition, coefficients):
        """Вычислить ошибки восстановленного источника.

        Parameters
        ----------
        condition : dict
            Условие с числом узлов базиса на интервале 0–3 часа.
        coefficients : array_like, shape (condition["nodes"],)
            Безразмерные узловые значения оценки.

        Returns
        -------
        scores : dict of str to float
            Метрики source_scores с масштабом Qref из конфигурации.

        See Also
        --------
        source_scores : Определения нормировок и единиц метрик.
        """

        basis = P1Basis(np.linspace(0,3,condition["nodes"]),time_unit="h")
        return source_scores(self.source,basis,coefficients,self.spec["Qref"])
