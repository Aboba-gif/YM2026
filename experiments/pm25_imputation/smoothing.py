"""Сглаживание Тихонова для рядов PM₂.₅ с шагом 20 минут."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def _inputs(values, weights, lam):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not y.size:
        raise ValueError("values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("values must be finite and nonnegative")
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must match values and be finite and strictly positive")
    parameter = np.asarray(lam, dtype=float)
    if parameter.ndim != 0 or not np.isfinite(parameter) or parameter < 0:
        raise ValueError("lam must be a finite nonnegative scalar")
    return y, w, float(parameter)


def _scaled_system(w, lam):
    """Разделить веса и lam на общий максимум перед сборкой трёхдиагональной системы."""
    scale = max(float(np.max(w)), lam)
    scaled_w = w / scale
    scaled_lam = lam / scale
    if np.any(scaled_w == 0):
        raise ValueError("weight/lambda dynamic range exceeds floating-point precision")
    diagonal = scaled_w.copy()
    if w.size > 1:
        diagonal[:-1] += scaled_lam
        diagonal[1:] += scaled_lam
    return scaled_w, scaled_lam, diagonal


def smooth_pm25(values, *, weights=None, lam=1.0) -> np.ndarray:
    r"""Сгладить ряд PM₂.₅ с квадратичным штрафом по первым разностям.

    Parameters
    ----------
    values : array_like, shape (n,)
        Непустой ряд конечных неотрицательных значений в единицах входа
        без пропусков.
    weights : array_like, shape (n,), optional
        Положительные конечные веса; по умолчанию все равны единице.
    lam : float, optional
        Неотрицательная сила сглаживания, по умолчанию 1. Безразмерна при
        безразмерных весах.

    Returns
    -------
    ndarray, shape (n,)
        Сглаженный ряд в единицах входа. При нулевом `lam` или одном отсчёте
        возвращается копия.

    Notes
    -----
    Для y = `values`, w = `weights` и λ = `lam` минимизируется функционал

    .. math::

        F(z) = \frac12 \sum_{i=1}^{n} w_i(z_i-y_i)^2
             + \frac{\lambda}{2} \sum_{i=1}^{n-1}(z_{i+1}-z_i)^2.

    Концы ряда свободны. Положительные веса задают единственный минимум
    в точной арифметике. Веса передаются в аргументе ``weights``; функция
    не оценивает дисперсию шума или интервалы неопределённости.
    """
    y, w, lam = _inputs(values, weights, lam)
    if lam == 0 or y.size == 1:
        return y.copy()
    scaled_w, scaled_lam, diagonal = _scaled_system(w, lam)
    band = np.zeros((3, y.size), dtype=float)
    band[0, 1:] = -scaled_lam
    band[1] = diagonal
    band[2, :-1] = -scaled_lam
    z = solve_banded((1, 1), band, scaled_w * y, check_finite=False)
    tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.max(y)))
    if not np.all(np.isfinite(z)) or np.min(z) < -tolerance:
        raise ArithmeticError("smoothing violated finite nonnegativity numerically")
    return np.maximum(z, 0.0)


def smoothing_diagnostics(values, smoothed, *, weights=None, lam=1.0) -> dict:
    """Вычислить функционал, шероховатость и невязку сглаженного ряда.

    Parameters
    ----------
    values : array_like, shape (n,)
        Конечный неотрицательный исходный ряд в единицах входа.
    smoothed : array_like, shape (n,)
        Проверяемый конечный неотрицательный сглаженный ряд в тех же единицах.
    weights : array_like, shape (n,), optional
        Положительные конечные веса функционала `smooth_pm25`; по умолчанию
        единицы.
    lam : float, optional
        Неотрицательная сила сглаживания функционала `smooth_pm25`.

    Returns
    -------
    dict
        Энергии и суммы квадратов разностей в квадрате единицы входа
        при безразмерных весах; экстремумы в единицах входа и безразмерная
        относительная невязка нормальных уравнений.

    Notes
    -----
    `normal_equation_relative_residual` — невязка системы, нормированная
    в норме ℓ∞ на сумму нормы матрицы, умноженной на норму `smoothed`,
    и нормы правой части. При нулевом знаменателе она равна нулю.
    Малая невязка характеризует решение системы, а не точность
    восстановления истинных концентраций.
    """
    y, w, lam = _inputs(values, weights, lam)
    z = np.asarray(smoothed, dtype=float)
    if z.shape != y.shape or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("smoothed must match values and be finite and nonnegative")
    scaled_w, scaled_lam, diagonal = _scaled_system(w, lam)
    rhs = scaled_w * y
    residual = diagonal * z - rhs
    if z.size > 1:
        residual[:-1] -= scaled_lam * z[1:]
        residual[1:] -= scaled_lam * z[:-1]
    row_norm = diagonal.copy()
    if z.size > 1:
        row_norm[:-1] += scaled_lam
        row_norm[1:] += scaled_lam
    denominator = float(np.max(row_norm) * np.max(np.abs(z)) + np.max(np.abs(rhs)))
    relative_residual = float(np.max(np.abs(residual)) / denominator) if denominator else 0.0
    data_energy = 0.5 * float(np.sum(w * (z - y) ** 2))
    roughness_before = float(np.sum(np.diff(y) ** 2))
    roughness_after = float(np.sum(np.diff(z) ** 2))
    return {
        "n_samples": int(y.size),
        "grid_step_minutes": 20,
        "lambda_dimensionless": lam,
        "data_energy": data_energy,
        "penalty_energy": 0.5 * lam * roughness_after,
        "objective": data_energy + 0.5 * lam * roughness_after,
        "objective_at_input": 0.5 * lam * roughness_before,
        "roughness_before": roughness_before,
        "roughness_after": roughness_after,
        "normal_equation_relative_residual": relative_residual,
        "scaled_weighted_sum_error": float(np.sum(scaled_w * (z - y))),
        "minimum": float(np.min(z)),
        "maximum": float(np.max(z)),
        "input_minimum": float(np.min(y)),
        "input_maximum": float(np.max(y)),
    }
