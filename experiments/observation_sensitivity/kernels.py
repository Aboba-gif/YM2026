"""Нормированные ядра на R², их вторые моменты и пространственная квадратура."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import expn


_I0 = float(expn(2, 1.0))
_I1 = _I0 - float(expn(3, 1.0))
_RADIUS_PER_WIDTH = math.sqrt(2.0 * _I0 / _I1)


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values (not bool/complex/object)")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value, name):
    result = _real_array(value, name)
    if result.ndim != 0 or float(result) <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(result)


def _width(value, compact=False):
    width = _positive_scalar(value, "width")
    scale = width * (_RADIUS_PER_WIDTH if compact else 1.0)
    square = scale * scale
    normalizer = math.pi * square * (_I0 if compact else 2.0)
    if (not math.isfinite(normalizer) or normalizer <= 0
            or not math.isfinite(1.0 / normalizer)):
        raise ValueError("width is outside the representable float64 kernel scale")
    return width


def _geometry(points, center):
    nodes = _real_array(points, "points")
    origin = _real_array(center, "center")
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) == 0:
        raise ValueError("points must have nonempty shape (nodes, 2)")
    if origin.shape != (2,):
        raise ValueError("center must have shape (2,)")
    # Разность противоположных больших координат может переполниться; они лежат вне представимого
    # хвоста ядра конечной ширины.
    with np.errstate(over="ignore"):
        return nodes - origin


def _owned(values):
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("kernel result is not representable in float64")
    result.flags.writeable = False
    return result


def _scaled_exp(exponent, area, normalizer, multiplier=None):
    """Вычислить масштабированную экспоненту с восстановлением крайних значений."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = area * np.exp(exponent) / normalizer
        if multiplier is not None:
            result *= multiplier
            # При нулевом множителе аналитическая производная нулевая, даже если ненужное
            # произведение плотности и площади переполняется.
            result[multiplier == 0] = 0.0
        repair = ((result == 0) | ~np.isfinite(result)) & np.isfinite(exponent)
        if multiplier is not None:
            repair &= np.isfinite(multiplier) & (multiplier != 0)
        log_value = math.log(area) - math.log(normalizer) + exponent[repair]
        if multiplier is not None:
            log_value += np.log(np.abs(multiplier[repair]))
        result[repair] = np.exp(log_value)
        if multiplier is not None:
            result[repair] *= np.sign(multiplier[repair])
    return result


@dataclass(frozen=True)
class GaussianKernel:
    """Изотропное гауссово ядро с единичной массой на плоскости.
    
    Parameters
    ----------
    width : float
        Положительное стандартное отклонение каждой координаты
        в тех же единицах длины, что и координаты точек.
    """

    width: float

    def __post_init__(self):
        object.__setattr__(self, "width", _width(self.width))

    @property
    def support_radius(self):
        """Вернуть признак неограниченного носителя ядра.

        Returns
        -------
        None
            Гауссово ядро имеет неограниченный носитель.
        """
        return None

    @property
    def coordinate_variance(self):
        """Вернуть дисперсию одной координаты ядра.

        Returns
        -------
        float
            Квадрат width в квадрате единиц длины.
        """

        return self.width**2

    def _evaluate(self, points, center, *, area=1.0, derivative=False):
        offset = _geometry(points, center)
        square = self.width**2
        with np.errstate(over="ignore", under="ignore"):
            distance2 = np.sum(offset**2, axis=1)
            ratio = distance2 / square
            # При переполнении r² отношение r²/width² вычисляется после масштабирования.
            extreme = ~np.isfinite(distance2)
            ratio[extreme] = np.sum((offset[extreme] / self.width)**2, axis=1)
            exponent = -distance2 / (2 * square)
            exponent[extreme] = -0.5 * ratio[extreme]
            multiplier = None
            if derivative:
                resolved = np.isfinite(ratio)
                multiplier = np.zeros_like(ratio)
                multiplier[resolved] = ratio[resolved] - 2.0
            result = _scaled_exp(exponent, area, 2 * np.pi * square, multiplier)
        return _owned(result)

    def density(self, points, center=(0.0, 0.0)):
        """Вычислить плотность ядра в заданных точках.

        Parameters
        ----------
        points : array_like, shape (n_points, 2)
            Непустой набор координат в тех же единицах длины, что и width.
        center : array_like, shape (2,), optional
            Центр ядра в тех же единицах; по умолчанию начало координат.

        Returns
        -------
        ndarray, shape (n_points,)
            Плотность в обратном квадрате единиц длины; массив доступен только
            для чтения.
        """
        return self._evaluate(points, center)

    def log_width_derivative(self, points, center=(0.0, 0.0)):
        r"""Вычислить производную плотности по логарифму ширины.

        Parameters
        ----------
        points : array_like, shape (n_points, 2)
            Непустой набор фиксированных координат в тех же единицах длины,
            что и width.
        center : array_like, shape (2,), optional
            Фиксированный центр ядра; по умолчанию начало координат.

        Returns
        -------
        ndarray, shape (n_points,)
            Производная в тех же единицах, что плотность; массив доступен только
            для чтения.

        Notes
        -----
        Для плотности :math:`\rho` и ширины :math:`s` вычисляется
        :math:`s\,\partial\rho/\partial s` при фиксированной геометрии.
        """
        return self._evaluate(points, center, derivative=True)


@dataclass(frozen=True)
class CompactKernel:
    """Гладкое ядро с круглым носителем и единичной массой на плоскости.
    
    Радиус носителя ``support_radius`` согласован со вторым моментом.
    Форма ядра и нормирующие интегралы приведены в README.
    
    Parameters
    ----------
    width : float
        Положительное стандартное отклонение каждой координаты,
        в тех же единицах длины, что и координаты точек.
    """

    width: float

    def __post_init__(self):
        object.__setattr__(self, "width", _width(self.width, compact=True))

    @property
    def support_radius(self):
        """Вернуть радиус круглого носителя ядра.

        Returns
        -------
        float
            Радиус в тех же единицах длины, что и width; согласован
            с координатной дисперсией.
        """

        return self.width * _RADIUS_PER_WIDTH

    @property
    def coordinate_variance(self):
        """Вернуть дисперсию одной координаты ядра.

        Returns
        -------
        float
            Квадрат width в квадрате единиц длины.
        """

        return self.width**2

    def _evaluate(self, points, center, *, area=1.0, derivative=False):
        offset = _geometry(points, center)
        radius = self.support_radius
        with np.errstate(over="ignore", under="ignore"):
            u = np.sum((offset / radius)**2, axis=1)
        inside = u < 1.0
        result = np.zeros(len(u), dtype=np.float64)
        # Внутри носителя gap > 0; на границе и снаружи плотность и производная нулевые.
        
        gap = 1.0 - u[inside]
        multiplier = -2.0 + 2.0 * u[inside] / gap**2 if derivative else None
        values = _scaled_exp(-1.0 / gap, area, np.pi * radius**2 * _I0, multiplier)
        result[inside] = values
        return _owned(result)

    def density(self, points, center=(0.0, 0.0)):
        """Вычислить плотность ядра в заданных точках.

        Parameters
        ----------
        points : array_like, shape (n_points, 2)
            Непустой набор координат в тех же единицах длины, что и width.
        center : array_like, shape (2,), optional
            Центр ядра в тех же единицах; по умолчанию начало координат.

        Returns
        -------
        ndarray, shape (n_points,)
            Плотность в обратном квадрате единиц длины; массив доступен только
            для чтения.
        """
        return self._evaluate(points, center)

    def log_width_derivative(self, points, center=(0.0, 0.0)):
        r"""Вычислить производную плотности по логарифму ширины.

        Parameters
        ----------
        points : array_like, shape (n_points, 2)
            Непустой набор фиксированных координат в тех же единицах длины,
            что и width.
        center : array_like, shape (2,), optional
            Фиксированный центр ядра; по умолчанию начало координат.

        Returns
        -------
        ndarray, shape (n_points,)
            Производная в тех же единицах, что плотность; массив доступен только
            для чтения.

        Notes
        -----
        Для плотности :math:`\rho` и ширины :math:`s` вычисляется
        :math:`s\,\partial\rho/\partial s` при фиксированной геометрии.
        """
        return self._evaluate(points, center, derivative=True)


def spatial_weights(kernel, points, centers, cell_area, *, derivative=False):
    """Построить пространственные квадратурные веса наблюдений.
    
    Parameters
    ----------
    kernel : GaussianKernel or CompactKernel
        Нормированное на плоскости ядро.
    points : array_like, shape (n_points, 2)
        Координаты узлов сетки в тех же единицах длины, что и ширина ядра.
    centers : array_like, shape (n_centers, 2)
        Координаты постов в тех же единицах.
    cell_area : float
        Положительная площадь ячейки в квадрате единицы координат.
    derivative : bool, optional
        Дифференцировать по логарифму ширины при фиксированной геометрии.
    
    Returns
    -------
    ndarray, shape (n_centers, n_points)
        Безразмерные веса: плотность в узле, умноженная на площадь ячейки,
        или их производные. Порядок входов сохраняется; ограничение
        областью не сопровождается повторной нормировкой.
    """
    if not isinstance(kernel, (GaussianKernel, CompactKernel)):
        raise TypeError("kernel must be GaussianKernel or CompactKernel")
    if type(derivative) is not bool:
        raise TypeError("derivative must be bool")
    area = _positive_scalar(cell_area, "cell_area")
    receivers = _real_array(centers, "centers")
    if receivers.ndim != 2 or receivers.shape[1] != 2 or len(receivers) == 0:
        raise ValueError("centers must have nonempty shape (receivers, 2)")
    return _owned(np.vstack([
        kernel._evaluate(points, center, area=area, derivative=derivative)
        for center in receivers
    ]))
