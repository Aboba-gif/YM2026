"""Признаки PM₂.₅ по видимым соседним постам и контексту пропуска."""
import numpy as np
import pandas as pd
from timeseries.features import feature_from_function


def network_features(inputs, target_index):
    """Вычислить сетевые и календарные признаки целевого поста.

    Parameters
    ----------
    inputs : MaskedInputs
        Видимые значения четырёх постов с одним каналом ``pm25``.
    target_index : int
        Индекс целевого поста.

    Returns
    -------
    FeatureValues
        Три концентрации соседей, три маски доступности и семь календарных
        признаков в целевом столбце постов. Остальные посты имеют NaN.
        Единица концентраций — неподтверждённая единица исходного экспорта.
    """
    neighbors = [j for j in range(4) if j != target_index]
    names = tuple(f"neighbor_{j}_{kind}" for kind in ("value", "available") for j in neighbors)
    names += ("annual_sin", "annual_cos", "diurnal_sin", "diurnal_cos", "month", "hour", "weekday")
    units = ("raw_export_unit",) * 3 + ("1",) * 8 + ("h", "1")

    def transform(visible):
        index = pd.DatetimeIndex(visible.times)
        values = visible.values[:, neighbors, 0]
        hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
        annual = 2 * np.pi * (index.dayofyear.to_numpy() - 1) / 365.2425
        diurnal = 2 * np.pi * hour / 24.0
        matrix = np.column_stack([
            values, np.isfinite(values).astype(float),
            np.sin(annual), np.cos(annual), np.sin(diurnal), np.cos(diurnal),
            index.month.to_numpy(), hour, index.dayofweek.to_numpy(),
        ])
        result = np.full((len(index), 4, len(names)), np.nan)
        result[:, target_index, :] = matrix
        return result

    return feature_from_function(names=names, units=units, transform=transform,
                                  uses_future=False).transform(inputs)


def common_features(inputs, start, end, target_index):
    """Вычислить признаки скрытого блока по видимому контексту окна.

    Parameters
    ----------
    inputs : MaskedInputs
        Окно четырёх постов с одним каналом ``pm25``. Целевой блок скрыт,
        оба его соседних конца доступны.
    start, end : int
        Локальные полуоткрытые границы блока.
    target_index : int
        Индекс целевого поста.

    Returns
    -------
    FeatureValues
        Двадцать признаков в целевом блоке. Вне блока и у остальных постов
        значения равны NaN. Используется ретроспективный контекст, включая
        правый конец блока. Физическая единица экспорта не подтверждена.
    """
    neighbors = [j for j in range(4) if j != target_index]
    names = tuple(f"neighbor_{j}_{kind}" for kind in ("value", "available") for j in neighbors)
    names += ("left", "right", "target_mean", "target_std", "target_fraction")
    names += tuple(f"neighbor_{j}_mean" for j in neighbors)
    names += tuple(f"neighbor_{j}_fraction" for j in neighbors)
    names += ("gap_steps", "linear", "relative_position")
    units = ("raw_export_unit",) * 3 + ("1",) * 3
    units += ("raw_export_unit",) * 4 + ("1",) + ("raw_export_unit",) * 3
    units += ("1",) * 3 + ("step", "raw_export_unit", "1")

    def transform(visible):
        window = visible.values[:, [target_index, *neighbors], 0]
        if not np.isnan(window[start:end, 0]).all():
            raise ValueError("Hide the full target gap before feature construction")
        n = end - start
        left, right = window[start - 1, 0], window[end, 0]
        observed_target = window[np.isfinite(window[:, 0]), 0]
        target_summary = [np.mean(observed_target) if len(observed_target) else np.nan,
                          np.std(observed_target) if len(observed_target) else np.nan,
                          len(observed_target) / len(window)]
        neighbors_values = window[start:end, 1:]
        means = [np.mean(window[np.isfinite(window[:, j]), j])
                 if np.isfinite(window[:, j]).any() else np.nan for j in range(1, 4)]
        fractions = np.isfinite(window[:, 1:]).mean(axis=0)
        context = np.tile([left, right, *target_summary, *means, *fractions, n], (n, 1))
        linear = np.linspace(left, right, n + 2)[1:-1]
        matrix = np.column_stack([neighbors_values, np.isfinite(neighbors_values).astype(float), context,
                                  linear, np.arange(1, n + 1) / (n + 1)])
        result = np.full((len(window), 4, len(names)), np.nan)
        result[start:end, target_index, :] = matrix
        return result

    return feature_from_function(names=names, units=units, transform=transform,
                                  uses_future=True).transform(inputs, context="retrospective")
