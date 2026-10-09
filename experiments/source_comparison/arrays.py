"""Преобразование научных входов в конечные вещественные массивы."""
import numpy as np


def finite_real_array(value, name):
    """Создать независимую копию конечных вещественных значений.

    Parameters
    ----------
    value : array_like
        Значения, преобразуемые в float. Логические значения и числовые
        строки допускаются; форма сохраняется.
    name : str
        Название входа для сообщения об ошибке.

    Returns
    -------
    array : ndarray
        Собственный доступный для записи массив float без NaN и бесконечностей.

    Raises
    ------
    ValueError
        Комплексные или неконечные значения либо ошибка преобразования.
    TypeError
        Значения не поддерживают преобразование в float.
    """
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result
