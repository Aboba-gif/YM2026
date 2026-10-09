"""Аналитические профили источника для генерации наблюдений и оценки ошибки.

Время измеряется в часах, интенсивность источника — в C·км²/ч,
где C обозначает единицу концентрации модели.
Заданный интеграл профиля относится к интервалу от 0 до 3 часов.
"""
from dataclasses import asdict, dataclass

import numpy as np

from experiments.source_comparison.truth import (
    BiExponential, FiniteRelease, build_truth,
)


def _positive(value):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError("positive finite scalar mass/scale required")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("positive finite scalar mass/scale required")
    return result


def _interval(source, left, right):
    values = np.asarray([left, right])
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("finite real integration endpoints required")
    left, right = map(float, values)
    if not source.start_hours <= left <= right <= source.end_hours:
        raise ValueError("interval outside declared source clock")
    return left, right


@dataclass(frozen=True)
class TwoPulse:
    """Аналитический профиль двух неперекрывающихся импульсов.

    Parameters
    ----------
    shape : {'jump', 'cosine'}
        Прямоугольные импульсы или импульсы приподнятого косинуса.
    mass : float, optional
        Положительный интеграл на 0–3 часах в C·км²; C — единица
        концентрации.
    start_hours, end_hours : float, optional
        Закреплённые границы области определения: -0.5 и 3 часа.

    Notes
    -----
    На обоих краях каждого импульса value возвращает ноль. Для cosine
    продолжение нулём имеет непрерывную первую производную, но не вторую;
    для jump значения разрывны на краях.
    """
    shape: str
    mass: float = 100.
    start_hours: float = -.5
    end_hours: float = 3.

    def __post_init__(self):
        if self.shape not in ("jump", "cosine"):
            raise ValueError("shape must be jump or cosine")
        object.__setattr__(self, "mass", _positive(self.mass))
        if self.start_hours != -.5 or self.end_hours != 3.:
            raise ValueError("the registered two-pulse clock is [-.5,3] hours")

    @property
    def components(self):
        """Вернуть параметры двух импульсов.

        Returns
        -------
        components : tuple of tuple of float
            Для каждого импульса: центр в часах, ширина в часах и безразмерная
            доля общего интеграла.
        """

        return ((5/7, 1/3, .6), (13/7, .5, .4))

    @property
    def events(self):
        """Вернуть границы событий аналитического профиля.

        Returns
        -------
        events : tuple of float
            Границы импульсов в часах, используемые для разбиения интегралов.
        """

        return tuple(edge for center, width, _ in self.components
                     for edge in (center-width/2, center+width/2))

    def value(self, hours):
        """Вычислить интенсивность источника в заданные моменты.

        Parameters
        ----------
        hours : array_like
            Конечные физические времена в часах в пределах start_hours и
            end_hours.

        Returns
        -------
        values : ndarray
            Интенсивности в C·км²/ч с формой входа; при скалярном времени
            возвращается массив нулевой размерности.
        """

        if np.iscomplexobj(hours):
            raise ValueError("source time must be real")
        t = np.asarray(hours, dtype=float)
        if not np.isfinite(t).all() or np.any(t < self.start_hours) or np.any(t > self.end_hours):
            raise ValueError("time outside declared source clock")
        result = np.zeros_like(t)
        for center, width, fraction in self.components:
            active = (t > center-width/2) & (t < center+width/2)
            shape = 1. if self.shape == "jump" else 1+np.cos(2*np.pi*(t-center)/width)
            result += self.mass*fraction/width*active*shape
        return result

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

        left, right = _interval(self, left, right)
        result = 0.
        for center, width, fraction in self.components:
            a, b = max(left, center-width/2), min(right, center+width/2)
            if a >= b:
                continue
            integral = b-a
            if self.shape == "cosine":
                k = 2*np.pi/width
                integral += (np.sin(k*(b-center))-np.sin(k*(a-center)))/k
            result += self.mass*fraction/width*integral
        return float(result)


@dataclass(frozen=True)
class ScaledSource:
    """Аналитический профиль источника с масштабированием интенсивности.

    Parameters
    ----------
    source : FiniteRelease, BiExponential or TwoPulse
        Исходный зарегистрированный профиль.
    scale : float
        Положительный безразмерный множитель значений и интегралов.
        Известная предыстория масштабируется тем же множителем.
    """
    source: FiniteRelease | BiExponential | TwoPulse
    scale: float

    def __post_init__(self):
        if not isinstance(self.source, (FiniteRelease, BiExponential, TwoPulse)):
            raise TypeError("only registered analytic source types are supported")
        object.__setattr__(self, "scale", _positive(self.scale))

    @property
    def start_hours(self):
        """Вернуть границу времени исходного профиля.

        Returns
        -------
        start_hours : float
            Начало области определения исходного профиля в часах.
        """

        return self.source.start_hours

    @property
    def end_hours(self):
        """Вернуть границу времени исходного профиля.

        Returns
        -------
        end_hours : float
            Конец области определения исходного профиля в часах.
        """

        return self.source.end_hours

    @property
    def events(self):
        """Вернуть точки разбиения исходного аналитического профиля.

        Returns
        -------
        events : tuple of float
            Точки событий в часах; пустой кортеж для профиля без событий.
        """

        return self.source.events

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
            Интенсивности в C·км²/ч с формой входа; C — единица концентрации
            модели.
        """

        return self.scale*self.source.value(hours)

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

        return self.scale*self.source.integral(left, right)


SOURCE_PROFILE_IDS = {"PG10": "SF01-PG10", "SB150": "SF01-SB150",
                  "EC04": "SF03-EC04", "EC06": "SF03-EC06"}


def make_sources(protocol_binding, *, mass=100.):
    """Создать шесть аналитических профилей с заданным интегралом.

    Parameters
    ----------
    protocol_binding : ProtocolBinding
        Протокол с четырьмя исходными профилями источника.
    mass : float, optional
        Положительный интеграл каждого профиля от 0 до 3 часов, в C·км².
        Тем же множителем масштабируется его известная предыстория.

    Returns
    -------
    sources : dict of str to ScaledSource
        Четыре профиля из протокола и два двухимпульсных профиля по их именам.
    """
    mass = _positive(mass)
    result = {}
    for name, source_id in SOURCE_PROFILE_IDS.items():
        base = build_truth(protocol_binding, source_id).source
        if not np.isclose(base.integral(0., 3.), 100., atol=2e-12, rtol=0):
            raise ValueError("Source profiles must have the declared mass 100")
        result[name] = ScaledSource(base, mass/100.)
    result["NEW-J2"] = ScaledSource(TwoPulse("jump"), mass/100.)
    result["NEW-S2"] = ScaledSource(TwoPulse("cosine"), mass/100.)
    return result


def first_moment(source, left, right):
    """Вычислить первый временной момент интенсивности источника.

    Parameters
    ----------
    source : FiniteRelease, BiExponential, TwoPulse or ScaledSource
        Зарегистрированный аналитический профиль.
    left, right : float
        Границы интегрирования в часах в пределах области определения.

    Returns
    -------
    moment : float or numpy.float64
        Интеграл произведения физического времени и интенсивности, в
        C·км²·ч.
    """
    left, right = _interval(source, left, right)
    if isinstance(source, ScaledSource):
        return source.scale*first_moment(source.source, left, right)
    if isinstance(source, FiniteRelease):
        a, b = max(left, source.events[0]), min(right, source.events[1])
        return 0. if a >= b else source.amplitude*(b-a)*(a+b)/2
    if isinstance(source, BiExponential):
        z = (right-left)/np.asarray(source.decay_hours)
        return float(sum(a*d*np.exp(-left/d)*(left*(-np.expm1(-width)) +
                         d*(-np.expm1(-width)-width*np.exp(-width)))
                         for a, d, width in zip(source.amplitudes, source.decay_hours, z)))
    if isinstance(source, TwoPulse):
        result = 0.
        for center, width, fraction in source.components:
            a, b = max(left, center-width/2), min(right, center+width/2)
            if a >= b:
                continue
            integral = (b-a)*(a+b)/2
            if source.shape == "cosine":
                k = 2*np.pi/width
                primitive = lambda t: t*np.sin(k*(t-center))/k+np.cos(k*(t-center))/k**2
                integral += primitive(b)-primitive(a)
            result += source.mass*fraction/width*integral
        return float(result)
    raise TypeError("unregistered analytic source")


def unknown_L2_squared(source):
    """Вычислить квадрат нормы L² источника на трёх часах.

    Parameters
    ----------
    source : FiniteRelease, BiExponential, TwoPulse or ScaledSource
        Зарегистрированный профиль на интервале от 0 до 3 часов.

    Returns
    -------
    squared_norm : float or numpy.float64
        Интеграл квадрата интенсивности, в C²·км⁴/ч.
    """
    if isinstance(source, ScaledSource):
        return source.scale**2*unknown_L2_squared(source.source)
    if isinstance(source, FiniteRelease):
        return source.amplitude*source.integral(0., 3.)
    if isinstance(source, BiExponential):
        return float(sum(a*b*(-np.expm1(-3*(1/d+1/e)))/(1/d+1/e)
                         for a, d in zip(source.amplitudes, source.decay_hours)
                         for b, e in zip(source.amplitudes, source.decay_hours)))
    if isinstance(source, TwoPulse):
        factor = 1. if source.shape == "jump" else 1.5
        return float(factor*source.mass**2*sum(p*p/w for _, w, p in source.components))
    raise TypeError("unregistered analytic source")


def source_record(source):
    """Составить запись параметров масштабированного источника.

    Parameters
    ----------
    source : ScaledSource
        Масштабированный зарегистрированный аналитический профиль.

    Returns
    -------
    record : dict
        Исходный профиль, множитель, единицы и интегралы известной
        предыстории и неизвестной части в C·км².
    """
    if not isinstance(source, ScaledSource):
        raise TypeError("record an explicitly scaled registered source")
    return dict(type=type(source.source).__name__, definition=asdict(source.source),
                scale=source.scale, unknown_mass=source.integral(0., 3.),
                known_history_mass=source.integral(-.5, 0.),
                time_unit="hour", source_unit="concentration*km^2/hour",
                status="conditional synthetic generator, not measured emissions")
