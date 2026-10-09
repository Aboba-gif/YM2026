"""Аналитические профили источника для синтетических наблюдений.

Время измеряется в часах, интенсивность источника — в C·км²/ч,
где C обозначает единицу концентрации модели.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from adrkit.config.validation import JSONRecord
from adrkit.errors import ConfigError


def _times(value, start, end):
    if np.iscomplexobj(value):
        raise ValueError("truth times must be real")
    t = np.asarray(value, dtype=float)
    if not np.isfinite(t).all() or np.any(t < start) or np.any(t > end):
        raise ValueError("truth requested outside its declared physical clock")
    return t


def _interval(left, right, start, end):
    left, right = _times([left, right], start, end)
    if left > right:
        raise ValueError("integral endpoints must be ordered")
    return float(left), float(right)


@dataclass(frozen=True)
class FiniteRelease:
    """Прямоугольный выброс конечной длительности.

    C обозначает единицу концентрации модели. Интенсивность включает левую
    границу выброса и исключает правую.

    Parameters
    ----------
    onset_hours : float
        Начало выброса в часах.
    duration_hours : float
        Положительная длительность выброса в часах.
    amplitude : float
        Положительная интенсивность внутри выброса в C·км²/ч.
    start_hours, end_hours : float, optional
        Границы области определения в часах; выброс целиком лежит внутри
        неё.
    """

    onset_hours: float
    duration_hours: float
    amplitude: float
    start_hours: float = -0.5
    end_hours: float = 3.

    def __post_init__(self):
        values = [self.onset_hours, self.duration_hours, self.amplitude,
                  self.start_hours, self.end_hours]
        if (not np.isfinite(values).all() or self.amplitude <= 0
                or self.duration_hours <= 0 or not self.start_hours < self.end_hours
                or self.onset_hours < self.start_hours
                or self.onset_hours+self.duration_hours > self.end_hours):
            raise ValueError("invalid finite release parameters")

    @property
    def events(self):
        """Вернуть границы событий аналитического профиля.

        Returns
        -------
        events : tuple of float
            Границы импульсов в часах, используемые для разбиения интегралов.
        """

        return (self.onset_hours, self.onset_hours+self.duration_hours)

    def value(self, hours):
        """Вычислить интенсивность источника в заданные моменты.

        Parameters
        ----------
        hours : array_like
            Конечные физические времена в часах в пределах start_hours и
            end_hours.

        Returns
        -------
        values : ndarray or numpy.float64
            Интенсивности в C·км²/ч с формой входа; при скалярном времени
            возвращается скаляр NumPy.
        """

        t = _times(hours, self.start_hours, self.end_hours)
        return self.amplitude*((t >= self.events[0]) & (t < self.events[1]))

    def integral(self, left, right):
        """Вычислить интеграл интенсивности на заданном интервале.

        Parameters
        ----------
        left, right : float
            Упорядоченные границы в часах внутри области определения профиля.

        Returns
        -------
        integral : float or numpy.float64
            Интеграл источника в C·км²; равен нулю на интервале нулевой длины.
        """

        left, right = _interval(left, right, self.start_hours, self.end_hours)
        return self.amplitude*max(0., min(right, self.events[1])-max(left, self.events[0]))


@dataclass(frozen=True)
class BiExponential:
    """Аналитический профиль источника как сумма двух затухающих экспонент.

    Parameters
    ----------
    amplitudes : tuple of float
        Две положительные интенсивности при нулевом времени в C·км²/ч.
    decay_hours : tuple of float
        Два положительных времени затухания в часах.
    start_hours, end_hours : float, optional
        Границы области определения профиля в часах.
    """
    amplitudes: tuple
    decay_hours: tuple
    start_hours: float = -0.5
    end_hours: float = 3.

    def __post_init__(self):
        a, d = np.asarray(self.amplitudes), np.asarray(self.decay_hours)
        if (a.shape != (2,) or d.shape != (2,)
                or not np.isfinite([*a, *d, self.start_hours, self.end_hours]).all()
                or np.any(a <= 0) or np.any(d <= 0)
                or not self.start_hours < self.end_hours):
            raise ValueError("two positive finite amplitudes and decay times required")
        object.__setattr__(self, "amplitudes", tuple(float(v) for v in a))
        object.__setattr__(self, "decay_hours", tuple(float(v) for v in d))

    @property
    def events(self):
        """Вернуть границы событий аналитического профиля.

        Returns
        -------
        events : tuple of float
            Пустой кортеж: профиль непрерывен без событий разбиения.
        """

        return ()

    def value(self, hours):
        """Вычислить интенсивность источника в заданные моменты.

        Parameters
        ----------
        hours : array_like
            Конечные физические времена в часах в пределах start_hours и
            end_hours.

        Returns
        -------
        values : ndarray or numpy.float64
            Интенсивности в C·км²/ч с формой входа; при скалярном времени
            возвращается скаляр NumPy.
        """

        t = _times(hours, self.start_hours, self.end_hours)
        return sum(a*np.exp(-t/d) for a, d in zip(self.amplitudes, self.decay_hours))

    def integral(self, left, right):
        """Вычислить интеграл интенсивности на заданном интервале.

        Parameters
        ----------
        left, right : float
            Упорядоченные границы в часах внутри области определения профиля.

        Returns
        -------
        integral : float or numpy.float64
            Интеграл источника в C·км²; равен нулю на интервале нулевой длины.
        """

        left, right = _interval(left, right, self.start_hours, self.end_hours)
        return sum(a*d*np.exp(-left/d)*(-np.expm1(-(right-left)/d))
                   for a, d in zip(self.amplitudes, self.decay_hours))


def _finite_release(generator, qref, start, end):
    return FiniteRelease(float(Fraction(generator["onset_min"]))/60,
                         generator["duration_min"]/60,
                         qref*float(Fraction(generator["amplitude_over_Qref"])), start, end)


def _bi_exponential(generator, qref, start, end):
    f1, d1, f2, d2 = generator["published_F1_d1min_F2_d2min"]
    d = np.array([d1, d2], dtype=float)
    f_at_zero = np.array([f1, f2])*np.exp(-generator["age_min"]/d)
    j_minutes = np.sum(f_at_zero*d*(-np.expm1(-generator["window_min"]/d)))
    return BiExponential(tuple(qref*60*f_at_zero/j_minutes), tuple(d/60), start, end)


# Зарегистрированные аналитические семейства задают свои формулы и параметры.
_BUILDERS = {"SF01": _finite_release, "SF03": _bi_exponential}


@dataclass(frozen=True)
class BoundTruth:
    """Аналитический профиль и его привязка к протоколу.

    Parameters
    ----------
    source : FiniteRelease or BiExponential
        Синтетический источник с физическим временем в часах.
    provenance : JSONRecord
        Идентификатор, параметры генератора, единицы и хеши протокола.
    """

    source: FiniteRelease | BiExponential
    provenance: JSONRecord


def build_truth(binding, source_id):
    """Создать аналитический источник из протокола сравнения.

    Parameters
    ----------
    binding : ProtocolBinding
        Протокол с зарегистрированными генераторами и их хешами.
    source_id : str
        Идентификатор источника в разделе source_strata.

    Returns
    -------
    truth : BoundTruth
        Профиль с параметрами, единицами и привязкой к протоколу.

    Raises
    ------
    ConfigError
        Источник или его семейство не зарегистрированы либо хеш записи не
        совпадает.
    """
    protocol = binding.document.to_dict()
    records = {r["id"]: r for r in protocol["source_strata"]["required"]}
    hashes = {r["id"]: r["sha256"]
              for r in protocol["source_strata"]["generator_spec_bindings"]}
    if source_id not in records:
        raise ConfigError(f"source is not explicitly registered: {source_id}")
    record = records[source_id]
    digest = JSONRecord(record).sha256
    if digest != hashes.get(source_id):
        raise ConfigError("source generator specification hash mismatch")
    family = record["family_id"]
    if family not in _BUILDERS:
        raise ConfigError(f"no implementation for source family: {family}")
    config = protocol["execution_config"]
    source = _BUILDERS[family](record["generator"], config["basis"]["Qref"],
        config["model"]["extended_start_hours"], config["model"]["extended_end_hours"])
    return BoundTruth(source, JSONRecord({
        "source_id": source_id, "spec_sha256": digest,
        "protocol_sha256": binding.full_sha256, "source_record": record,
        "time_unit": "hour", "source_unit": "C*km^2/hour",
        "status": "synthetic source with prescribed analytical profile",
    }))


def interval_controls(source, state_times, *, origin):
    """Вычислить среднюю интенсивность источника на шагах решателя.

    Parameters
    ----------
    source : FiniteRelease, BiExponential, TwoPulse or ScaledSource
        Профиль с методами ``value(hours)`` и ``integral(left, right)``.
    state_times : array_like, shape (n_steps + 1,)
        Возрастающие границы шагов в часах, начиная с нуля.
    origin : float
        Физическое время начала сетки в часах.

    Returns
    -------
    controls : ndarray, shape (n_steps + 1,)
        Массив той же длины, что `state_times`: начальное значение и затем
        средние интенсивности в C·км²/ч
        на последовательных временных интервалах.
    """
    times = np.asarray(state_times)
    if (np.iscomplexobj(times) or times.ndim != 1 or len(times) < 2
            or not np.isfinite(times).all() or times[0] != 0
            or np.any(np.diff(times) <= 0) or not np.isfinite(origin)):
        raise ValueError("finite increasing state clock starting at zero required")
    edges = times+origin
    controls = np.array([source.value(origin), *(
        source.integral(float(left), float(right))/width
        for left, right, width in zip(edges[:-1], edges[1:], np.diff(times)))], dtype=float)
    if controls.shape != times.shape or not np.isfinite(controls).all():
        raise ValueError("source must return finite scalar values and integrals")
    return controls
