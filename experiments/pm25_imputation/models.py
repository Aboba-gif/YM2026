"""Кандидаты заполнения PM₂.₅ по видимым исходным наблюдениям.

Методы получают исходную панель, при проверке — её копию со скрытым блоком.
Ранее заполненные значения не используются ни в признаках, ни как обучающие
метки. Выбор метода и оценка ошибок выполняются отдельно."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from timeseries.contracts import MaskedInputs

from .features import common_features, network_features
from .panel import hide_inputs, window_inputs
from .data import SITES


DEFAULT_CONFIG = {
    "hgb": {
        "loss": "squared_error",
        "max_leaf_nodes": 7,
        "max_iter": 100,
        "learning_rate": 0.06,
        "l2_regularization": 2.0,
        "min_samples_leaf": 40,
        "early_stopping": False,
        "random_state": 20260907,
    },
    "common_window_steps": 288,
    "common_stride": 144,
    "common_lengths": [1, 3, 12, 36, 72, 216],
    "common_max_windows": 300,
}


def reorder(values: np.ndarray, target_index: int) -> np.ndarray:
    """Переставить целевой пост в первый столбец отдельной копии панели.

    Parameters
    ----------
    values : ndarray, shape (n, 4)
        Исходные концентрации PM₂.₅ в единицах исходного экспорта в порядке `SITES`; NaN обозначает
        пропуск.
    target_index : int
        Индекс целевого столбца от 0 до 3.

    Returns
    -------
    ndarray, shape (n, 4)
        Копия с целевым постом первым; порядок остальных постов сохранён.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Expected a four-station panel")
    if not isinstance(target_index, (int, np.integer)) or not 0 <= target_index < 4:
        raise ValueError("target_index must be one of 0, 1, 2, 3")
    columns = [target_index, *(j for j in range(4) if j != target_index)]
    return values[:, columns].copy()


def _checked_panel(index, values):
    index = pd.DatetimeIndex(index)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape != (len(index), 4) or len(index) == 0:
        raise ValueError("Expected a nonempty four-station panel aligned with index")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Index must be unique, increasing, and finite")
    if len(index) > 1 and not np.all(index[1:] - index[:-1] == pd.Timedelta(minutes=20)):
        raise ValueError("The retrospective panel must use a regular 20-minute grid")
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("Original PM2.5 values must be nonnegative finite numbers or NaN")
    return index, values




def _runs(mask: np.ndarray):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


class CoreModels:
    """Кандидаты заполнения PM₂.₅ с отдельными этапами обучения и предсказания.

    Метки берутся только из исходных наблюдений.
    `fit_record` сохраняет настройки и состав обучающих окон.

    Parameters
    ----------
    config : dict or None, optional
        Переопределения ``DEFAULT_CONFIG``: параметры ``hgb`` и размеры окон
        в отсчётах по 20 минут. Словарь ``hgb`` объединяется с настройками
        по умолчанию.
    """

    def __init__(self, config: dict | None = None):
        self.config = deepcopy(DEFAULT_CONFIG)
        for key, value in (config or {}).items():
            if key not in self.config:
                raise ValueError(f"Unknown CoreModels configuration key: {key}")
            if key == "hgb":
                self.config[key].update(value)
            else:
                self.config[key] = deepcopy(value)
        cfg = self.config
        lengths = cfg["common_lengths"]
        if not lengths or any(not isinstance(n, int) or n < 1 for n in lengths):
            raise ValueError("common_lengths must contain positive integer lengths")
        if sorted(set(lengths)) != lengths:
            raise ValueError("common_lengths must be sorted and unique")
        for key in ("common_window_steps", "common_stride", "common_max_windows"):
            if not isinstance(cfg[key], int) or cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if cfg["common_window_steps"] < max(lengths) + 2:
            raise ValueError("Common windows must contain the gap and two endpoints")
        if cfg["hgb"]["early_stopping"] is not False:
            raise ValueError("Fixed training requires early_stopping=False")
        if cfg["hgb"]["loss"] != "squared_error":
            raise ValueError("Final-export candidates use raw-scale squared_error")
        self.fit_record = None
        self.common_model = None
        self.network_model = None

    def fit(self, inputs, train_mask: np.ndarray, target_index: int = 0):
        """Обучить кандидатов заполнения на исходных наблюдениях.

        Parameters
        ----------
        inputs : MaskedInputs
            Исходные видимые значения четырёх постов с одним каналом
            ``pm25``. Физическая единица экспорта не подтверждена.
        train_mask : ndarray of bool, shape (n,)
            Маска обучающих отсчётов. Окна для ``common_hgb`` целиком лежат
            в непрерывных участках этой маски.
        target_index : int, optional
            Индекс целевого поста, по умолчанию 0.

        Returns
        -------
        CoreModels
            Этот же объект с обученными моделями и записью `fit_record`.

        Notes
        -----
        Для сетевой модели каждое наблюдаемое обучающее значение используется
        с фактическими соседями и с полностью скрытыми соседями. Модель
        ``common_hgb`` обучается на искусственно скрытых блоках внутри
        выбранных окон.
        """
        index, values = self._checked_inputs(inputs, target_index)
        self.target_index = target_index
        train_mask = np.asarray(train_mask)
        if train_mask.dtype != np.bool_ or train_mask.shape != (len(index),):
            raise ValueError("train_mask must be a Boolean array aligned with index")
        use = train_mask & np.isfinite(values[:, 0])
        if not use.any():
            raise ValueError("Cannot fit without originally observed training targets")

        x = network_features(inputs, target_index).values[use, target_index, :]
        y = values[use, 0]
        outage_x = x.copy()
        outage_x[:, :3] = np.nan
        outage_x[:, 3:6] = 0.0
        augmented_x = np.vstack([x, outage_x])
        # Колонки без конечных обучающих значений исключаются только по train_mask.
        self.network_columns = np.flatnonzero(np.isfinite(augmented_x).any(axis=0))
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.network_model = HistGradientBoostingRegressor(**self.config["hgb"])
        self.network_model.fit(augmented_x[:, self.network_columns], np.tile(y, 2))
        self.global_mean = float(np.mean(y))
        self.climatology = {}
        months, hours = index.month.to_numpy(), index.hour.to_numpy()
        for month in range(1, 13):
            for hour in range(24):
                group = use & (months == month) & (hours == hour)
                if group.any():
                    self.climatology[(month, hour)] = float(np.mean(values[group, 0]))

        cfg = self.config
        width, lengths = cfg["common_window_steps"], cfg["common_lengths"]
        candidates, eligible = [], []
        for a, b in _runs(train_mask):
            for start in range(int(a), int(b) - width + 1, cfg["common_stride"]):
                candidates.append(start)
                if any(np.isfinite(values[start + (width - length) // 2 - 1:
                                          start + (width - length) // 2 + length + 1, 0]).all()
                       for length in lengths):
                    eligible.append(start)
        selected = eligible
        if len(selected) > cfg["common_max_windows"]:
            positions = np.linspace(0, len(selected) - 1, cfg["common_max_windows"], dtype=int)
            selected = [selected[i] for i in positions]
        feature_parts, label_parts, support = [], [], []
        unique_targets = np.zeros(len(index), dtype=bool)
        for window_start in selected:
            original = values[window_start:window_start + width]
            window = window_inputs(inputs, window_start, window_start + width)
            for length in lengths:
                start = (width - length) // 2
                end = start + length
                if not np.isfinite(original[start - 1:end + 1, 0]).all():
                    continue
                truth = original[start:end, 0].copy()
                hidden = np.zeros(window.values.shape, dtype=bool)
                hidden[start:end, target_index, 0] = True
                visible = hide_inputs(window, mit_mask=hidden)
                feature_parts.append(common_features(visible, start, end, target_index)
                                     .values[start:end, target_index, :])
                label_parts.append(truth)
                outage = visible.visible.copy()
                outage[:, target_index, :] = False
                visible = hide_inputs(visible, artificial_mask=outage)
                feature_parts.append(common_features(visible, start, end, target_index)
                                     .values[start:end, target_index, :])
                label_parts.append(truth)
                unique_targets[window_start + start:window_start + end] = True
                support.append(dict(window_start=window_start, length=length,
                                    target_occurrences=2 * length))
        self.common_model = None
        self.common_columns = np.array([], dtype=int)
        if feature_parts:
            common_x = np.vstack(feature_parts)
            self.common_columns = np.flatnonzero(np.isfinite(common_x).any(axis=0))
            self.common_model = HistGradientBoostingRegressor(**cfg["hgb"])
            self.common_model.fit(common_x[:, self.common_columns], np.concatenate(label_parts))
        self.fit_record = {
            "config": deepcopy(cfg),
            "label_source": "originally observed target values only",
            "network_input": "three original neighbor PM2.5 values, masks, raw-clock calendar",
            "network_original_training_labels": int(use.sum()),
            "network_augmented_target_occurrences": int(2 * use.sum()),
            "network_retained_feature_indices": self.network_columns.tolist(),
            "climatology_nonempty_month_hour_groups": len(self.climatology),
            "climatology_global_mean": self.global_mean,
            "common_candidate_windows": len(candidates),
            "common_eligible_windows": len(eligible),
            "common_selected_window_starts": list(selected),
            "common_training_support": support,
            "common_unique_original_targets": int(unique_targets.sum()),
            "common_augmented_target_occurrences": sum(r["target_occurrences"] for r in support),
            "common_available": self.common_model is not None,
            "common_retained_feature_indices": self.common_columns.tolist(),
            "feature_filter": "omit columns entirely nonfinite in augmented training inputs only",
            "training_first_observed_target": str(index[np.flatnonzero(use)[0]]),
            "training_last_observed_target": str(index[np.flatnonzero(use)[-1]]),
            "common_augmentation": ["actual", "all_neighbors_missing_entire_window"],
            "network_augmentation": ["actual", "all_neighbors_missing"],
        }
        return self

    def common_window_bounds(self, n_samples: int, start: int, end: int):
        """Определить границы окна по действующей конфигурации модели.

        Parameters
        ----------
        n_samples : int
            Число отсчётов в полной панели.
        start, end : int
            Границы пропуска: 0 <= start < end <= n_samples;
            конец не включается.

        Returns
        -------
        bounds : tuple of int or None
            Начало и невключаемый конец окна. None означает, что панель
            короче окна или пропуск длиннее max(common_lengths).

        Notes
        -----
        Определяет только геометрию. Наличие обученной модели ``common_hgb`` и наблюдений
        на концах пропуска проверяет ``predict_gap``.
        """
        width = self.config["common_window_steps"]
        if n_samples < width or end - start > max(self.config["common_lengths"]):
            return None
        window_start = int(np.clip(start - (width - (end - start)) // 2,
                                   0, n_samples - width))
        return window_start, window_start + width

    @staticmethod
    def _checked_inputs(inputs, target_index):
        if not isinstance(inputs, MaskedInputs):
            raise ValueError("Models require MaskedInputs without evaluation truth")
        layout = inputs.spec.to_dict()
        if layout["sites"] != list(SITES) or layout["channels"] != ["pm25"]:
            raise ValueError("Models require the four PM25 stations in SITES order")
        return _checked_panel(inputs.times, reorder(inputs.values[:, :, 0], target_index))

    def predict_gap(self, inputs, start: int, end: int, target_index: int = 0, *, method=None):
        """Предсказать скрытый блок целевого поста.

        Parameters
        ----------
        inputs : MaskedInputs
            Видимые значения четырёх постов с одним каналом ``pm25``;
            целевой блок полностью скрыт. Единицы исходного экспорта
            сохраняются; физическая единица не подтверждена.
        start, end : int
            Полуоткрытые границы блока в пределах входов.
        target_index : int, optional
            Индекс обученного целевого поста, по умолчанию 0.
        method : {'linear', 'common_hgb', 'network_hgb', 'climatology'} or None, optional
            Один кандидат либо все четыре, если задано None.

        Returns
        -------
        ndarray or None or dict
            Конечный неотрицательный вектор выбранного метода или None
            при его неприменимости. Без ``method`` — словарь четырёх
            кандидатов в порядке linear, common_hgb, network_hgb, climatology.

        Notes
        -----
        Линейная интерполяция требует видимых значений на обоих концах.
        Для ``common_hgb`` дополнительно нужны обученная модель, окно
        установленной ширины и допустимая длина пропуска.
        """
        if self.fit_record is None:
            raise RuntimeError("Fit CoreModels before prediction")
        index, values = self._checked_inputs(inputs, target_index)
        if target_index != self.target_index:
            raise ValueError("Prediction target differs from the fitted station")
        if not (isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer))
                and 0 <= start < end <= len(index)):
            raise ValueError("Gap must satisfy 0 <= start < end <= len(index)")
        if not np.isnan(values[start:end, 0]).all():
            raise ValueError("Conceal every original target in the full gap before prediction")
        result = dict(linear=None, common_hgb=None, network_hgb=None, climatology=None)
        if method is not None and method not in result:
            raise ValueError("Unknown PM25 method")
        if method in (None, "network_hgb"):
            gap = window_inputs(inputs, start, end)
            features = network_features(gap, target_index).values[:, target_index, :]
            result["network_hgb"] = np.maximum(0.0, self.network_model.predict(features[:, self.network_columns]))
        if method in (None, "climatology"):
            result["climatology"] = np.asarray([
                self.climatology.get((int(t.month), int(t.hour)), self.global_mean)
                for t in index[start:end]], dtype=float)
        bracketed = start > 0 and end < len(values) and np.isfinite(values[[start - 1, end], 0]).all()
        if bracketed and method in (None, "linear"):
            result["linear"] = np.linspace(values[start - 1, 0], values[end, 0], end - start + 2)[1:-1]
        if bracketed and self.common_model is not None and method in (None, "common_hgb"):
            bounds = self.common_window_bounds(len(values), start, end)
            if bounds is not None:
                window_start, window_end = bounds
                local_start, local_end = start - window_start, end - window_start
                if local_start > 0 and local_end < window_end - window_start:
                    window = window_inputs(inputs, window_start, window_end)
                    x = common_features(window, local_start, local_end, target_index).values[
                        local_start:local_end, target_index, :]
                    result["common_hgb"] = np.maximum(0.0, self.common_model.predict(x[:, self.common_columns]))
        for name, predicted in result.items():
            if predicted is not None and (predicted.shape != (end - start,) or not np.isfinite(predicted).all()
                                          or (predicted < 0).any()):
                raise RuntimeError(f"{name} produced an invalid prediction")
        return result if method is None else result[method]
