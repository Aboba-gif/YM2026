"""Численная сборка E06 с использованием общего решателя ADR."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adrkit.backends.cached import CachedSolver
from adrkit.config.validation import JSONRecord
from adrkit.loads import IntervalAverageLoad, disclose_history
from experiments.source_comparison.truth import interval_controls
from adrkit.observations.grid import GridObservation
from adrkit.predictions import SelectedPrediction
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace
from experiments.source_recovery.backend import (
    make_solver, source_scores,
)

from .design import FULL_TIMES_HOURS, PRIMARY_TICKS, ObservationSpec, PathSpec
from .kernels import CompactKernel, GaussianKernel, spatial_weights
from .operators import (
    ProjectedPrediction, solver_state_space, temporal_weights,
)


PRIMARY_ROWS = tuple(station*72+tick for station in range(4) for tick in PRIMARY_TICKS)
DEFAULT_OBSERVATIONS = (
    ObservationSpec("gaussian", 1.), ObservationSpec("gaussian", .5),
    ObservationSpec("gaussian", 2.), ObservationSpec("compact", 1.),
    ObservationSpec("gaussian", 1., "average", 1/3),
)


def _owned(values):
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _coefficients(values):
    raw = np.asarray(values)
    if raw.shape != (73,) or raw.dtype.kind not in "iuf":
        raise ValueError("Coefficients must be a real numeric vector of length 73")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Coefficients must be finite and nonnegative")
    return result


def _signal(values):
    value = np.asarray(values)
    if value.shape != (288,) or not np.isfinite(value).all():
        raise ValueError("Projection must be a finite 288-coordinate signal")
    return _owned(value)


def _residual(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Solver residual certificate must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class PreparedEstimator:
    """Оценщик с P1-базисом и линеаризацией в нулевой точке.

    Parameters
    ----------
    dense_prediction : ProjectedPrediction
        Прогноз 288 наблюдений с известной предысторией источника.
    restricted_prediction : SelectedPrediction
        Прогноз 36 наблюдений для восстановления.
    basis : P1Basis
        Базис с 73 узлами на интервале 0–3 ч.
    jacobian : ndarray, shape (36, 73)
        Якобиан прогноза по безразмерным коэффициентам; единицы концентрации.
    offset : ndarray, shape (36,)
        Прогноз при нулевых коэффициентах в единицах концентрации.
    """
    dense_prediction: ProjectedPrediction
    restricted_prediction: SelectedPrediction
    basis: P1Basis
    jacobian: np.ndarray
    offset: np.ndarray


class ProductionBackend:
    """Модель E06 для прогноза наблюдений и оценки ошибки источника.

    Конфигурация копируется при создании. Устройство линеаризации и кешей
    описано в README.

    Parameters
    ----------
    spec : dict
        Научная конфигурация E06: сетки, часы, геометрия постов и масштаб
        Qref.
    source : object
        Временной источник с методами value и integral; интенсивность в
        C·км²/ч, где C — единица концентрации модели.
    solver_factory : callable, optional
        Фабрика solver_factory(spec_copy, grid_name, reaction_gamma),
        возвращающая CachedSolver.
    source_id : str or None, optional
        Идентификатор источника; при None задаётся первым запросом PathSpec.
    """

    def __init__(self, spec, source, *, solver_factory=make_solver, source_id=None):
        self._record = JSONRecord(spec)
        self._spec = self._record.to_dict()
        if not callable(solver_factory):
            raise TypeError("solver_factory must be callable")
        if any(not callable(getattr(source, name, None)) for name in ("value", "integral")):
            raise TypeError("source must expose value and integral methods")
        if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
            raise ValueError("source_id must be a nonempty string or None")
        model = self._spec["model"]
        if model["origin_hours"] != -.5 or model["horizon"] != 3.5:
            raise ValueError("E06 clock is physical [-.5,3] hours")
        qref = self._spec["Qref"]
        if isinstance(qref, bool) or not isinstance(qref, (int, float)) or qref <= 0:
            raise ValueError("Qref must be a positive finite scalar")
        centers = np.asarray(self._spec["observations"]["centers_km"])
        if centers.shape != (4, 2) or centers.dtype.kind not in "iuf" or not np.isfinite(centers).all():
            raise ValueError("Exactly four finite real station coordinates are required")
        self._source, self._source_id, self._factory = source, source_id, solver_factory
        self._basis = P1Basis(np.linspace(0., 3., 73), time_unit="h")
        self._solvers, self._operators, self._loads = {}, {}, {}
        self._truth, self._initial = {}, {}
        self._counts = dict(solver_builds=0, truth_forward_attempts=0,
            truth_forward_calls=0, initializer_batches_attempted=0,
            scoring_forward_attempts=0)

    def _check_path(self, path):
        if not isinstance(path, PathSpec):
            raise TypeError("Explicit PathSpec required")
        if self._source_id is None:
            self._source_id = path.source
        elif self._source_id != path.source:
            raise ValueError("Path source differs from the backend's bound source_id")

    def _solver(self, grid_name):
        if not isinstance(grid_name, str) or grid_name not in self._spec["grids"]:
            raise ValueError("Requested grid must be explicitly declared in spec")
        if grid_name not in self._solvers:
            solver = self._factory(self._record.to_dict(), grid_name,
                                   self._spec["model"]["reaction_gamma"])
            if not isinstance(solver, CachedSolver):
                raise TypeError("solver_factory must return CachedSolver")
            self._solvers[grid_name] = solver.snapshot()
            self._counts["solver_builds"] += 1
        return self._solvers[grid_name]

    def _operator(self, observation, grid_name, derivative=False):
        if not isinstance(observation, ObservationSpec):
            raise TypeError("Explicit ObservationSpec required")
        if type(derivative) is not bool:
            raise TypeError("log_width_derivative must be bool")
        key = grid_name, observation, derivative
        if key not in self._operators:
            solver = self._solver(grid_name)
            kernel = (GaussianKernel if observation.spatial_kind == "gaussian" else CompactKernel)(
                observation.width_km)
            spatial = spatial_weights(kernel, solver.xy,
                self._spec["observations"]["centers_km"], solver.cell_area, derivative=derivative)
            temporal = temporal_weights(solver.times, FULL_TIMES_HOURS,
                origin_hours=self._spec["model"]["origin_hours"],
                kind=observation.temporal_kind, window_hours=observation.window_hours)
            state_space = solver_state_space(solver, state_unit="ug/m^3")
            # Идентификаторы совпадают с restricted_metric исходной серии и PanelFactory E06.
            # Совместимость определяется также идентификаторами координат.
            values = ArraySpace((288,), axes=("observation",),
                coordinates=(tuple(f"dense_row_{i}" for i in range(288)),), units=state_space.units)
            self._operators[key] = GridObservation(spatial, temporal,
                domain=state_space, codomain=values)
        return self._operators[key]

    def observation_weights(self, observation, *, grid_name="G0", log_width_derivative=False):
        """Вернуть пространственные и временные веса наблюдения.

        Parameters
        ----------
        observation : ObservationSpec
            Пространственное ядро и временной способ наблюдения.
        grid_name : str, optional
            Имя сетки в конфигурации; по умолчанию G0.
        log_width_derivative : bool, optional
            Вернуть производную пространственных весов по логарифму ширины при
            фиксированных координатах.

        Returns
        -------
        space_weights : ndarray, shape (4, n_nodes)
            Безразмерные веса постов в порядке координат сетки.
        time_weights : ndarray, shape (72, n_states)
            Безразмерные веса времён наблюдения в порядке состояний.
        """
        operator = self._operator(observation, grid_name, log_width_derivative)
        return operator.space_weights, operator.time_weights

    def _load(self, grid_name):
        if grid_name not in self._loads:
            solver = self._solver(grid_name)
            history = disclose_history(self._source, solver.times, self._spec["model"]["origin_hours"])
            known = np.asarray((*history.integrals, history.initial))
            if not np.isfinite(known).all() or np.any(known < 0):
                raise ValueError("Disclosed history must be finite and nonnegative")
            # IntervalAverageLoad вычисляет известную историю и хранит только числа. Генератор
            # неизвестного источника дальше не передаётся.
            self._loads[grid_name] = IntervalAverageLoad(self._basis, solver.times,
                origin=self._spec["model"]["origin_hours"], q_reference=self._spec["Qref"],
                history_integral=history.integral, history_initial=history.initial,
                source_unit="C*km^2/hour")
        return self._loads[grid_name]

    def initial_linearization(self, observation, *, grid_name="G0"):
        """Линеаризовать прогноз наблюдений при нулевых коэффициентах.

        Parameters
        ----------
        observation : ObservationSpec
            Пространственное ядро и временной способ наблюдения.
        grid_name : str, optional
            Имя сетки в конфигурации; по умолчанию G0.

        Returns
        -------
        jacobian : ndarray, shape (288, 73)
            Отдельная копия якобиана в единицах концентрации на единицу
            коэффициента.
        offset : ndarray, shape (288,)
            Отдельная копия свободного члена в единицах концентрации; включает
            известную предысторию источника.
        """
        self._operator(observation, grid_name)
        key = grid_name, observation
        if key not in self._initial:
            requested = tuple(dict.fromkeys((*DEFAULT_OBSERVATIONS, observation)))
            missing = tuple(h for h in requested if (grid_name, h) not in self._initial)
            operators = tuple(self._operator(h, grid_name) for h in missing)
            prediction = ProjectedPrediction(self._solver(grid_name), self._load(grid_name), operators[0])
            zero = np.zeros(73)
            self._counts["initializer_batches_attempted"] += 1
            try:
                matrices = prediction.projected_jacobians(zero, operators)
                updates = {}
                for h, operator, matrix in zip(missing, operators, matrices):
                    if matrix.shape != (288, 73) or not np.isfinite(matrix).all():
                        raise ValueError("Initializer Jacobian must be finite and have shape (288,73)")
                    updates[grid_name, h] = (_owned(matrix), _signal(prediction.project(zero, operator)))
                self._initial.update(updates)
            finally:
                prediction.invalidate()
        matrix, offset = self._initial[key]
        return _owned(matrix), _owned(offset)

    def prepare(self, path):
        """Подготовить оценщик для заданных условий восстановления.

        Parameters
        ----------
        path : PathSpec
            Условия источника, сетки, наблюдений и регуляризации.

        Returns
        -------
        PreparedEstimator
            Прогнозы, P1-базис и линеаризация для 36 выбранных наблюдений.
        """
        self._check_path(path)
        matrix, offset = self.initial_linearization(path.inverse_h, grid_name=path.grid)
        prediction = ProjectedPrediction(self._solver(path.grid), self._load(path.grid),
                                          self._operator(path.inverse_h, path.grid))
        if prediction.codomain.shape != (288,):
            raise ValueError("This study requires 288 dense observation coordinates")
        rows = np.array(PRIMARY_ROWS)
        return PreparedEstimator(prediction, SelectedPrediction(prediction, PRIMARY_ROWS), self._basis,
                                 _owned(matrix[rows]), _owned(offset[rows]))

    def prime_truth(self, *, grid_name="G0", observations=None,
                    include_log_width_derivatives=True):
        """Заполнить кеш проекций сигнала заданного источника.

        Сохраняются только отсутствующие проекции. Уже заполненный кеш
        используется повторно.

        Parameters
        ----------
        grid_name : str, optional
            Имя сетки в конфигурации; по умолчанию G0.
        observations : iterable of ObservationSpec or None, optional
            Непустой набор операторов; None выбирает DEFAULT_OBSERVATIONS.
        include_log_width_derivatives : bool, optional
            Дополнительно сохранить производные проекций по логарифму ширины.
        """
        if type(include_log_width_derivatives) is not bool:
            raise TypeError("include_log_width_derivatives must be bool")
        requested = DEFAULT_OBSERVATIONS if observations is None else tuple(observations)
        if not requested or any(not isinstance(h, ObservationSpec) for h in requested):
            raise ValueError("A nonempty sequence of ObservationSpec is required")
        derivatives = (False, True) if include_log_width_derivatives else (False,)
        keys = tuple(dict.fromkeys((grid_name, h, derivative)
                                   for h in requested for derivative in derivatives))
        missing = tuple(key for key in keys if key not in self._truth)
        operators = tuple(self._operator(h, grid, derivative) for grid, h, derivative in missing)
        if not missing:
            return
        solver = self._solver(grid_name)
        controls = interval_controls(self._source, solver.times, origin=self._spec["model"]["origin_hours"])
        if not np.isfinite(controls).all() or np.any(controls < 0):
            raise ValueError("True interval controls must be finite and nonnegative")
        self._counts["truth_forward_attempts"] += 1
        trajectory = solver.solve_controls(controls)
        self._counts["truth_forward_calls"] += 1
        residual = _residual(trajectory.max_scaled_residual)
        updates = {key: (_signal(operator.predict(trajectory.states)), residual)
                   for key, operator in zip(missing, operators)}
        self._truth.update(updates)
        # В кэш записаны только проекции и невязка; траектория состояния в нём не хранится.

    def project_truth(self, observation, *, grid_name="G0", log_width_derivative=False):
        """Вернуть проекцию заданного источника и невязку прямой задачи.

        Parameters
        ----------
        observation : ObservationSpec
            Пространственное ядро и временной способ наблюдения.
        grid_name : str, optional
            Имя сетки в конфигурации; по умолчанию G0.
        log_width_derivative : bool, optional
            Заменить проекцию её производной по логарифму ширины ядра.

        Returns
        -------
        signal : ndarray, shape (288,)
            Отдельная копия сигнала в мкг/м³, сначала по постам, затем по
            времени; без добавленного шума.
        residual : float
            Максимальная масштабированная невязка прямого решателя; безразмерная.
        """
        self._operator(observation, grid_name, log_width_derivative)
        key = grid_name, observation, log_width_derivative
        if key not in self._truth:
            self.prime_truth(grid_name=grid_name,
                observations=tuple(dict.fromkeys((*DEFAULT_OBSERVATIONS, observation))))
        signal, residual = self._truth[key]
        return _owned(signal), residual

    def truth(self, path):
        """Вернуть сигнал источника под порождающим оператором наблюдения.

        Parameters
        ----------
        path : PathSpec
            Условия источника, сетки, наблюдений и регуляризации.

        Returns
        -------
        signal : ndarray, shape (288,)
            Сигнал без добавленного шума в мкг/м³ на truth_grid.
        residual : float
            Максимальная безразмерная масштабированная невязка прямой задачи.
        """
        self._check_path(path)
        return self.project_truth(path.true_h, grid_name=path.truth_grid)

    def score_source(self, path, coefficients):
        """Вычислить ошибки восстановленного временного источника.

        Parameters
        ----------
        path : PathSpec
            Условия источника, сетки, наблюдений и регуляризации.
        coefficients : array_like, shape (73,)
            Конечные неотрицательные безразмерные коэффициенты P1-базиса на
            интервале 0–3 ч.

        Returns
        -------
        dict
            Ошибки профиля и интегральной интенсивности; ключи, единицы и
            нормировки определены в source_scores.
        """
        self._check_path(path)
        point = _coefficients(coefficients)
        return source_scores(self._source, self._basis, point, self._spec["Qref"])

    def score_prediction(self, path, coefficients):
        """Спрогнозировать наблюдения для восстановленного источника.

        Parameters
        ----------
        path : PathSpec
            Условия источника, сетки, наблюдений и регуляризации.
        coefficients : array_like, shape (73,)
            Конечные неотрицательные безразмерные коэффициенты P1-базиса на
            интервале 0–3 ч.

        Returns
        -------
        assumed : ndarray, shape (288,)
            Прогноз в мкг/м³ с предполагаемым оператором наблюдения.
        true_h : ndarray, shape (288,)
            Проекция того же восстановленного поля в мкг/м³ с порождающим
            оператором.
        residual : float
            Максимальная безразмерная масштабированная невязка прямой задачи.
        """
        self._check_path(path)
        point = _coefficients(coefficients)
        prediction = ProjectedPrediction(self._solver(path.grid), self._load(path.grid),
                                          self._operator(path.inverse_h, path.grid))
        true_operator = self._operator(path.true_h, path.grid)
        self._counts["scoring_forward_attempts"] += 1
        try:
            assumed = _signal(prediction.predict(point))
            actual = _signal(prediction.project(point, true_operator))
            return assumed, actual, _residual(prediction.trajectory.max_scaled_residual)
        finally:
            prediction.invalidate()

    @property
    def cache_info(self):
        """Вернуть счётчики расчётов и размеры кеша.

        Returns
        -------
        dict
            Завершённые truth_forward_calls, начатые *_attempts и объём массивов
            кеша в байтах. Объём кеша не равен памяти всего процесса.
        """
        return dict(self._counts, source_id=self._source_id,
            solver_grids=list(self._solvers), truth_projections=len(self._truth),
            initializer_matrices=len(self._initial),
            initializer_shapes=[list(matrix.shape) for matrix, _ in self._initial.values()],
            retained_trajectory_count=0,
            projection_cache_bytes=sum(signal.nbytes for signal, _ in self._truth.values()),
            initializer_cache_bytes=sum(matrix.nbytes+offset.nbytes for matrix, offset in self._initial.values()))
