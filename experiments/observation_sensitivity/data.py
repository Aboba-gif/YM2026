"""Парные данные E06 и фиксированная оценка рабочей ковариации."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from adrkit.config.validation import JSONRecord
from experiments.source_comparison.calibration import (
    CalibrationFailure, array_hash, covariance_guards, fit_covariance,
    noise_covariance,
)
from adrkit.inverse.misfit import covariance_metric
from adrkit.spaces import ArraySpace
from experiments.source_recovery import config as recovery_config, panels as recovery_panels

from .covariance import DEFAULT_GUARDS, ExponentialMixture, mixture_covariance
from .design import POPULATION_LENGTH_HOURS, PathSpec


def _owned(values):
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _signal(values):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.shape != (288,):
        raise ValueError("signal288 must be a real numeric vector of length 288")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("signal288 must be finite")
    return result


@dataclass(frozen=True)
class EstimationData:
    """Наблюдения подгонки и выбора с рабочей ковариационной метрикой.

    Parameters
    ----------
    fit : ndarray, shape (36,)
        Наблюдения подгонки в мкг/м³.
    selection : ndarray, shape (36,)
        Наблюдения выбора в мкг/м³ с отдельной шумовой выборкой.
    metric : CovarianceMetric
        Метрика выбранных координат наблюдения.
    selected_covariance : ndarray, shape (36, 36)
        Рабочая ковариация в квадрате единиц концентрации.
    provenance : JSONRecord
        Происхождение выборок, калибровки и ковариации.
    """
    fit: np.ndarray
    selection: np.ndarray
    metric: object
    selected_covariance: np.ndarray
    provenance: JSONRecord


@dataclass(frozen=True)
class TestData:
    """Проверочные наблюдения и запись их происхождения.

    Parameters
    ----------
    values : ndarray, shape (36,)
        Проверочная выборка в мкг/м³.
    provenance : JSONRecord
        Происхождение проверочной выборки и выбранных координат.
    """
    values: np.ndarray
    provenance: JSONRecord


@dataclass(frozen=True)
class _Weight:
    selected_covariance: np.ndarray
    calibration: JSONRecord
    selected_guards: JSONRecord


@dataclass(frozen=True)
class _RejectedCalibration:
    # Сохраняется причина, не исключение с traceback, удерживающим временные данные.
    reason: str


class PanelFactory:
    """Фабрика шумовых выборок и рабочих ковариаций E06.

    Парные условия используют общие случайные ошибки. Подгонка, выбор,
    проверка и калибровка имеют отдельные шумовые выборки. Правила ключей и
    обработки отказов описаны в README.

    Parameters
    ----------
    spec : mapping
        Конфигурация с разделом calibration; сохраняются только фиксированные
        настройки калибровки E06.
    """

    def __init__(self, spec):
        # Сохраняются только настройки калибровки; их значения ниже сверяются с протоколом E06.
        self._settings_record = JSONRecord(spec["calibration"])
        settings = self._settings_record.to_dict()
        if (settings.get("panels") != {"calib_noise": 32}
                or settings.get("sd_bounds") != [.001, 100.]
                or settings.get("guards") != asdict(DEFAULT_GUARDS)
                or not np.array_equal(np.asarray(settings.get("ell_grid_hours")),
                                      np.geomspace(1/60, 1.5, 33))):
            raise ValueError("E06 requires the unchanged v2 32-panel settings, SD bounds, ell grid and guards")
        self._weights = {}
        self._generating_covariances = {}
        self._residuals = {}
        self._rows = np.array(recovery_config.PRIMARY_ROWS, dtype=int, copy=True)
        self._rows.flags.writeable = False
        self._layout = ArraySpace((36,), axes=("observation",),
            coordinates=(tuple(f"dense_row_{i}" for i in self._rows),), units="ug/m^3")

    @staticmethod
    def _check_path(path):
        if not isinstance(path, PathSpec):
            raise TypeError("Explicit PathSpec required")
        exponential = path.noise.family == "station_exponential"
        if (path.weight == "W03") != exponential:
            raise ValueError("W03 requires exponential noise; mixture controls require mixture noise")
        expected_version = 2 if exponential else 3
        if path.stream.seed != 20260926 or path.stream.version != expected_version:
            raise ValueError("E06 uses fixed stream 2 for exponential and stream 3 for mixture noise")

    @staticmethod
    def _noise_key(path):
        return path.noise, path.stream, path.replicate

    @staticmethod
    def _stream(path):
        return dict(seed=path.stream.seed, version=path.stream.version)

    @staticmethod
    def _exponential_spec(noise, *, population=False):
        return dict(type="station_exponential", station_sd=list(noise.station_sd),
                    ell_hours=([POPULATION_LENGTH_HOURS]*4 if population
                               else list(noise.lengths_hours)))

    def _generating(self, path):
        key = self._noise_key(path)
        if key not in self._generating_covariances:
            noise = path.noise
            if noise.family == "station_exponential":
                matrix = noise_covariance(recovery_config.DENSE_TIMES, self._exponential_spec(noise))
                covariance_guards(matrix, self._settings_record.to_dict()["guards"])
            else:
                mixture = ExponentialMixture(noise.fast_hours, noise.slow_hours, noise.fast_weight)
                # Первый лаг смеси должен совпасть с экспонентой заданной популяционной длины.
                first_lag = float(mixture.correlation(1/3))
                expected = math.exp(-(1/3)/POPULATION_LENGTH_HOURS)
                if not math.isclose(first_lag, expected, rel_tol=2e-15, abs_tol=0.):
                    raise ValueError("Mixture first lag does not match the declared population optimum")
                matrix = mixture_covariance(recovery_config.DENSE_TIMES, noise.station_sd, mixture)
            self._generating_covariances[key] = _owned(matrix)
        return self._generating_covariances[key]

    def _residual(self, path, panel, index=0):
        key = (*self._noise_key(path), panel, index)
        if key not in self._residuals:
            values, record = recovery_panels.draw_residual(self._stream(path), path.replicate,
                                             panel, self._generating(path), index)
            self._residuals[key] = (_owned(values), JSONRecord(record))
        return self._residuals[key]

    def _build_weight(self, path):
        settings = self._settings_record.to_dict()
        if path.weight == "W03":
            # Оценка на редкой сетке и перенос на плотную сохраняют порядок вычислений E05.
            spec = dict(calibration=settings, stream=self._stream(path),
                        noise={"corr": self._exponential_spec(path.noise)})
            dense, record = recovery_panels.calibrate_dense(spec, path.replicate, "corr", "W03")
            selected = dense[np.ix_(self._rows, self._rows)]
        elif path.weight == "W_mix_oracle":
            selected = self._generating(path)[np.ix_(self._rows, self._rows)]
            record = dict(family=path.weight, status="oracle",
                scientific_role="inaccessible true mixture covariance control",
                calibration_inputs_used=False, covariance_sha256=array_hash(selected))
        elif path.weight == "W_exp_population":
            # Проверить первый лаг смеси до использования фиксированной популяционной экспоненты.
            self._generating(path)
            dense = noise_covariance(recovery_config.DENSE_TIMES,
                                     self._exponential_spec(path.noise, population=True))
            selected = dense[np.ix_(self._rows, self._rows)]
            record = dict(family=path.weight, status="population",
                scientific_role="known-variance population exponential control; not a field estimate",
                calibration_inputs_used=False, ell_hours=[POPULATION_LENGTH_HOURS]*4,
                station_variances=[float(s*s) for s in path.noise.station_sd],
                covariance_sha256=array_hash(selected))
        else:
            panels, records = [], []
            for index in range(32):
                residual, provenance = self._residual(path, "calibration", index)
                panels.append(residual.reshape(4, 72)[:, recovery_config.PRIMARY_TICKS])
                records.append(provenance.to_dict())
            residuals = np.stack(panels)
            fitted = fit_covariance(residuals, recovery_config.DENSE_TIMES[recovery_config.PRIMARY_TICKS],
                                    family="W03", specification=settings)
            selected = fitted.covariance
            record = dict(family=path.weight, status="accepted",
                parent_coarse_calibration=fitted.provenance.to_dict(),
                calibration_inputs_used=True, calibration_shape=list(residuals.shape),
                panel_records=records, covariance_sha256=array_hash(selected))
        diagnostics = covariance_guards(selected, settings["guards"])
        return _Weight(_owned(selected), JSONRecord(record), JSONRecord(diagnostics))

    def _weight(self, path):
        key = (*self._noise_key(path), path.weight)
        if key not in self._weights:
            try:
                self._weights[key] = self._build_weight(path)
            except CalibrationFailure as error:
                self._weights[key] = _RejectedCalibration(str(error))
        value = self._weights[key]
        if isinstance(value, _RejectedCalibration):
            raise CalibrationFailure(value.reason)
        return value

    def _design(self):
        return dict(removed_primary_tick_indices=[], rows=self._rows.tolist(),
                    retained_count=36, mask="none")

    def estimation(self, path, signal288):
        """Сформировать данные подгонки и выбора.

        Parameters
        ----------
        path : PathSpec
            Условия источника, сетки, наблюдений и регуляризации.
        signal288 : array_like, shape (288,)
            Сигнал без шума в мкг/м³; сначала посты, внутри каждого — времена.

        Returns
        -------
        EstimationData
            По 36 наблюдений каждой выборки и рабочая метрика. Калибровка
            использует отдельные ошибки с нулевым средним.
        """
        self._check_path(path)
        signal = _signal(signal288)
        weight = self._weight(path)
        observations, panel_records = {}, {}
        for name in ("fit", "selection"):
            residual, record = self._residual(path, name)
            observations[name] = _owned((signal+residual)[self._rows])
            panel_records[name] = record.to_dict()
        metric = covariance_metric(weight.selected_covariance, layout=self._layout)
        provenance = dict(calibration=weight.calibration.to_dict(),
            design=self._design(), panel_records=panel_records,
            fit_y_sha256=array_hash(observations["fit"]),
            selection_y_sha256=array_hash(observations["selection"]),
            selected_covariance_sha256=array_hash(weight.selected_covariance),
            selected_covariance_guards=weight.selected_guards.to_dict())
        return EstimationData(observations["fit"], observations["selection"],
                              metric, _owned(weight.selected_covariance), JSONRecord(provenance))

    def test(self, path, signal288):
        """Сформировать проверочные наблюдения.

        Вызывающая программа должна закрепить выбор параметров до запроса этих
        данных. Сама фабрика стадию checkpoint не проверяет.

        Parameters
        ----------
        path : PathSpec
            Условия источника, сетки, наблюдений и регуляризации.
        signal288 : array_like, shape (288,)
            Сигнал без шума в мкг/м³ в порядке плотных наблюдений.

        Returns
        -------
        TestData
            36 наблюдений с отдельной проверочной шумовой выборкой.
        """
        self._check_path(path)
        signal = _signal(signal288)
        residual, record = self._residual(path, "test")
        values = _owned((signal+residual)[self._rows])
        return TestData(values, JSONRecord(dict(
            design=self._design(), panel_record=record.to_dict(),
            test_y_sha256=array_hash(values))))
