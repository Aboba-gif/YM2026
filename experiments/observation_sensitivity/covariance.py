"""Экспоненциальные смеси ковариаций с блоками по постам и численная диагностика."""
from dataclasses import asdict, dataclass
import math
from numbers import Integral

import numpy as np
from scipy.linalg import block_diag, solve_triangular

from experiments.source_comparison.calibration import covariance_guards


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values (not bool/complex/object)")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name, *, positive=True):
    result = _real_array(value, name)
    if result.ndim != 0 or (positive and float(result) <= 0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite scalar")
    return float(result)


@dataclass(frozen=True)
class CovarianceGuards:
    """Пороги численной проверки ковариации.

    Все пороги безразмерны.

    Parameters
    ----------
    relative_symmetry : float, optional
        Положительный порог относительного дефекта симметрии в спектральной
        норме, меньше единицы.
    cond2_max : float, optional
        Предельное число обусловленности в спектральной норме, не меньше
        единицы.
    cholesky_backward : float, optional
        Положительный порог относительной обратной ошибки разложения
        Холецкого в спектральной норме, меньше единицы.
    whitening_spectral : float, optional
        Положительный порог спектральной нормы дефекта отбеливания, меньше
        единицы.
    """

    relative_symmetry: float = 1e-12
    cond2_max: float = 1e10
    cholesky_backward: float = 1e-12
    whitening_spectral: float = 1e-8

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = _scalar(getattr(self, name), name)
            if name == "cond2_max":
                if value < 1:
                    raise ValueError("cond2_max must be >= 1")
            elif value >= 1:
                raise ValueError(f"{name} must be < 1")
            object.__setattr__(self, name, value)


DEFAULT_GUARDS = CovarianceGuards()


def _guard(matrix, guards):
    if not isinstance(guards, CovarianceGuards):
        raise TypeError("guards must be CovarianceGuards")
    result = _real_array(matrix, "covariance")
    if result.ndim != 2 or not len(result) or result.shape[0] != result.shape[1]:
        raise ValueError("covariance must be a nonempty square matrix")
    return result, covariance_guards(result, asdict(guards))


@dataclass(frozen=True)
class ExponentialMixture:
    """Смесь двух экспоненциальных корреляций во времени.

    Parameters
    ----------
    fast_hours : float, optional
        Положительная длина быстрой корреляции в часах.
    slow_hours : float, optional
        Длина медленной корреляции в часах, строго больше fast_hours.
    fast_weight : float, optional
        Безразмерная доля быстрой компоненты, строго между нулём и единицей.
    """

    fast_hours: float = 1 / 12
    slow_hours: float = 1.0
    fast_weight: float = 0.5

    def __post_init__(self):
        fast = _scalar(self.fast_hours, "fast_hours")
        slow = _scalar(self.slow_hours, "slow_hours")
        weight = _scalar(self.fast_weight, "fast_weight", positive=False)
        if not fast < slow:
            raise ValueError("fast_hours must be strictly less than slow_hours")
        if not 0 < weight < 1:
            raise ValueError("fast_weight must be strictly between zero and one")
        object.__setattr__(self, "fast_hours", fast)
        object.__setattr__(self, "slow_hours", slow)
        object.__setattr__(self, "fast_weight", weight)

    def correlation(self, lags_hours):
        """Вычислить корреляцию смеси для временных лагов.

        Parameters
        ----------
        lags_hours : array_like
            Непустой скаляр или массив конечных неотрицательных лагов в часах.

        Returns
        -------
        ndarray
            Безразмерная корреляция в форме входа; массив доступен только для
            чтения.
        """
        lags = _real_array(lags_hours, "lags_hours")
        if np.any(lags < 0) or not lags.size:
            raise ValueError("lags_hours must be nonempty and nonnegative")
        with np.errstate(over="ignore", under="ignore"):
            result = (self.fast_weight * np.exp(-lags / self.fast_hours)
                      + (1 - self.fast_weight) * np.exp(-lags / self.slow_hours))
        result = np.array(result, dtype=np.float64, copy=True)
        result.flags.writeable = False
        return result


def matched_lag_mixture(delta_hours, population_length_hours, *, fast_hours=1 / 12,
                        slow_hours=1.0):
    """Подобрать смесь с заданной корреляцией первого лага.

    Если корреляции неразличимы в float64, функция отклоняет параметры.

    Parameters
    ----------
    delta_hours : float
        Положительный первый лаг в часах.
    population_length_hours : float
        Длина заданной экспоненциальной корреляции в часах, строго между
        fast_hours и slow_hours.
    fast_hours : float, optional
        Положительная длина быстрой компоненты в часах.
    slow_hours : float, optional
        Положительная длина медленной компоненты в часах.

    Returns
    -------
    ExponentialMixture
        Смесь, совпадающая с заданной экспонентой на первом лаге.
    """
    delta = _scalar(delta_hours, "delta_hours")
    length = _scalar(population_length_hours, "population_length_hours")
    fast = _scalar(fast_hours, "fast_hours")
    slow = _scalar(slow_hours, "slow_hours")
    if not fast < length < slow:
        raise ValueError("population_length_hours must lie strictly between fast and slow")
    af, ag, ass = [math.exp(-delta / x) for x in (fast, length, slow)]
    if not 0 < ag < 1 or not af < ag < ass:
        raise ValueError("requested first-lag correlations are not resolved in float64")
    weight = (ass - ag) / (ass - af)
    return ExponentialMixture(fast, slow, weight)


def mixture_covariance(times_hours, station_sd, mixture, *, guards=DEFAULT_GUARDS):
    """Построить ковариацию смеси для независимых гауссовых ошибок постов.
    
    Parameters
    ----------
    times_hours : array_like, shape (n_times,)
        Строго возрастающие моменты в часах; не менее двух.
    station_sd : array_like, shape (n_stations,)
        Положительные стандартные отклонения концентрации.
    mixture : ExponentialMixture
        Временная корреляция, общая для постов.
    guards : CovarianceGuards, optional
        Пороги численной проверки ковариации.
    
    Returns
    -------
    ndarray, shape (n_stations * n_times, n_stations * n_times)
        Блочно-диагональная ковариация в квадрате единиц концентрации.
        Сначала упорядочены посты, внутри каждого — времена.
    """
    if not isinstance(mixture, ExponentialMixture):
        raise TypeError("mixture must be ExponentialMixture")
    times = _real_array(times_hours, "times_hours")
    sd = _real_array(station_sd, "station_sd")
    if (times.ndim != 1 or len(times) < 2
            or np.any(times[1:] <= times[:-1])):
        raise ValueError("times_hours must be strictly increasing with at least two values")
    if sd.ndim != 1 or not len(sd) or np.any(sd <= 0):
        raise ValueError("station_sd must be a nonempty positive vector")
    with np.errstate(over="ignore", under="ignore"):
        lags = np.abs(times[:, None] - times[None, :])
        variances = sd**2
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("station variances are outside float64 range")
    correlation = mixture.correlation(lags)
    matrix = block_diag(*[variance * correlation for variance in variances])
    result, _ = _guard(matrix, guards)
    result.flags.writeable = False
    return result


def _population_inputs(a, lag1, n):
    values = _real_array(a, "a")
    r = _scalar(lag1, "lag1", positive=False)
    if not values.size or np.any(values < 0) or np.any(values >= 1) or not 0 <= r < 1:
        raise ValueError("a and lag1 must be in [0,1)")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 2:
        raise ValueError("n must be an integer >= 2")
    return values, r, int(n)


def _population_result(result):
    if not np.isfinite(result).all():
        raise ValueError("population diagnostic is outside float64 range")
    if result.ndim == 0:
        return float(result)
    result.flags.writeable = False
    return result


def population_objective(a, lag1, n=9):
    r"""Вычислить популяционный гауссов критерий AR(1).

    Parameters
    ----------
    a : array_like
        Непустой скаляр или массив предполагаемых корреляций первого лага от
        нуля включительно до единицы исключительно.
    lag1 : float
        Истинная корреляция первого лага в том же диапазоне.
    n : int, optional
        Число равноотстоящих наблюдений, не меньше двух.

    Returns
    -------
    float or ndarray
        Безразмерный критерий в форме a; для скалярного входа — float.

    Notes
    -----
    Средние нулевые, дисперсии известны. Критерий равен удвоенному
    ожиданию отрицательного логарифма плотности без постоянного слагаемого.
    Для корреляции :math:`a`, истинного первого лага :math:`r` и числа
    наблюдений :math:`n`:

    .. math::

        F(a)=(n-1)\log(1-a^2)
        +\frac{n+(n-2)a^2-2(n-1)ar}{1-a^2}.
    """
    values, r, size = _population_inputs(a, lag1, n)
    denominator = (1 - values) * (1 + values)
    numerator = (1 - values)**2 + 2 * values * (1 - r)
    result = ((size - 1) * np.log1p(-values**2)
              + 1 + (size - 1) * numerator / denominator)
    return _population_result(np.asarray(result))


def population_derivative(a, lag1, n=9):
    r"""Вычислить производную популяционного критерия AR(1).

    Parameters
    ----------
    a : array_like
        Непустой скаляр или массив предполагаемых корреляций первого лага от
        нуля включительно до единицы исключительно.
    lag1 : float
        Истинная корреляция первого лага в том же диапазоне.
    n : int, optional
        Число равноотстоящих наблюдений, не меньше двух.

    Returns
    -------
    float or ndarray
        Безразмерная производная по a в форме входа; для скаляра — float.

    Notes
    -----
    Здесь :math:`r` обозначает аргумент `lag1`, :math:`n` — число
    наблюдений. Предпосылки совпадают с population_objective.

    .. math::

        F'(a)=\frac{2(n-1)(a-r)(1+a^2)}{(1-a^2)^2}.
    """
    values, r, size = _population_inputs(a, lag1, n)
    denominator = (1 - values) * (1 + values)
    result = 2 * (size - 1) * (values - r) * (1 + values**2) / denominator**2
    return _population_result(np.asarray(result))


def covariance_diagnostics(true_covariance, assumed_covariance, *, guards=DEFAULT_GUARDS):
    r"""Сравнить гауссовы ковариации по KL и дефекту отбеливания.

    Parameters
    ----------
    true_covariance, assumed_covariance : array_like, shape (n, n)
        Симметричные положительно определённые ковариации в одинаковых
        координатах и единицах; средние нулевые.
    guards : CovarianceGuards, optional
        Пороги численной проверки обеих ковариаций.

    Returns
    -------
    dict
        Безразмерные kl_true_to_assumed, whitening_spectral и
        whitening_frobenius, а также true_guards и assumed_guards.

    Notes
    -----
    KL направлена от истинной гауссовой модели к предполагаемой.
    Спектральная и фробениусова нормы вычисляются для :math:`B-I`, где
    :math:`B=L^{-1}\Sigma_{\mathrm{true}}L^{-\mathsf T}` и
    :math:`LL^{\mathsf T}=\Sigma_{\mathrm{assumed}}`. Здесь :math:`L` —
    нижний фактор Холецкого предполагаемой ковариации, :math:`I` —
    единичная матрица размера :math:`n\times n`.
    """
    true, true_report = _guard(true_covariance, guards)
    assumed, assumed_report = _guard(assumed_covariance, guards)
    if true.shape != assumed.shape:
        raise ValueError("true and assumed covariance shapes must match")
    true_factor = np.linalg.cholesky(true)
    assumed_factor = np.linalg.cholesky(assumed)
    relative_factor = solve_triangular(assumed_factor, true_factor, lower=True)
    eigenvalues = np.linalg.svd(relative_factor, compute_uv=False)**2
    if np.any(eigenvalues <= 0) or not np.isfinite(eigenvalues).all():
        raise ValueError("relative covariance spectrum is outside float64 range")
    delta = eigenvalues - 1.0
    # log1p сохраняет точность около единицы; при delta == -1 нужен log(lambda).
    
    terms = np.empty_like(delta)
    close = np.abs(delta) < 0.5
    terms[close] = delta[close] - np.log1p(delta[close])
    terms[~close] = delta[~close] - np.log(eigenvalues[~close])
    kl = float(0.5 * np.sum(terms))
    left = solve_triangular(assumed_factor, true, lower=True)
    whitened = solve_triangular(assumed_factor, left.T, lower=True).T
    defect = whitened - np.eye(len(true))
    spectral = float(np.linalg.norm(defect, 2))
    # Масштабирование защищает сумму квадратов от переполнения.
    
    largest = float(np.max(np.abs(defect)))
    frobenius = (0.0 if largest == 0 else
                 largest * math.sqrt(float(np.sum((defect / largest)**2))))
    if not all(math.isfinite(x) and x >= 0 for x in (kl, spectral, frobenius)):
        raise ValueError("covariance diagnostics are outside float64 range")
    return dict(kl_true_to_assumed=kl, whitening_spectral=spectral,
                whitening_frobenius=frobenius,
                true_guards=true_report, assumed_guards=assumed_report)
