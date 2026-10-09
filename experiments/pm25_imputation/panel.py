"""Панель исходных значений PM₂.₅ и входы со скрытыми наблюдениями."""
import numpy as np
import pandas as pd
from timeseries.contracts import EvaluationTruth, MaskedInputs, NetworkPanel, PanelSpec

from .data import SITES


def network_panel(index, values, *, row_present=None):
    """Связать значения четырёх постов с временной сеткой.

    Parameters
    ----------
    index : DatetimeIndex
        Непустая регулярная сетка с шагом 20 минут без часового пояса.
    values : ndarray, shape (n, 4)
        Исходные числовые значения в порядке ``SITES``; NaN означает пропуск.
        Физическая единица исходного экспорта не подтверждена.
    row_present : ndarray of bool, shape (n, 4), optional
        Наличие исходной строки каждого поста. По умолчанию все строки
        считаются присутствующими.

    Returns
    -------
    NetworkPanel
        Неизменяемая панель с одним каналом ``pm25``.
    """
    index = pd.DatetimeIndex(index)
    if (not len(index) or index.tz is not None or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing
            or (len(index) > 1 and not np.all(np.diff(index) == pd.Timedelta(minutes=20)))):
        raise ValueError("Expected a nonempty regular naive 20-minute grid")
    values = np.asarray(values, dtype=float)
    if values.shape != (len(index), len(SITES)):
        raise ValueError("Expected four station columns aligned with index")
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("Original PM25 values must be nonnegative finite numbers or NaN")
    present = np.ones(values.shape, dtype=bool) if row_present is None else np.asarray(row_present)
    observed = np.isfinite(values)
    spec = PanelSpec(dict(start=str(index[0]), stop=str(index[-1] + pd.Timedelta(minutes=20)),
                          frequency="20min", origin=str(index[0]), clock="naive", timezone=None,
                          calendar="proleptic_gregorian", sites=list(SITES), channels=["pm25"],
                          units={"pm25": "raw_export_unit"}))
    return NetworkPanel(spec, tuple(index), values[:, :, None], present[:, :, None],
                        observed[:, :, None], (present & ~observed)[:, :, None])


def visible_inputs(panel, *, protocol_id, population_id, population_kind, train_mask=None):
    """Подготовить входы панели, при обучении скрыв внешние значения.

    Parameters
    ----------
    panel : NetworkPanel
        Исходные значения и маски присутствия.
    protocol_id, population_id : str
        Идентификаторы протокола и выборки.
    population_kind : {'train', 'selection', 'test', 'export'}
        Назначение входов.
    train_mask : ndarray of bool, shape (n,), optional
        Доступные обучающие отсчёты. Наблюдения вне маски скрываются.

    Returns
    -------
    MaskedInputs
        Видимые значения без контрольных ответов.
    """
    hidden = np.zeros(panel.values.shape, dtype=bool)
    if train_mask is not None:
        train_mask = np.asarray(train_mask)
        if train_mask.dtype != np.bool_ or train_mask.shape != (len(panel.times),):
            raise ValueError("train_mask must be a Boolean array aligned with index")
        hidden = panel.original_observed & ~train_mask[:, None, None]
    return MaskedInputs(panel.spec, panel.times, np.where(hidden, np.nan, panel.values),
                        panel.row_present, panel.original_observed, hidden,
                        np.zeros_like(hidden), protocol_id, population_id, population_kind)


def _window_spec(record, start, end):
    """Сохранить привязку сетки при изменении границ временного диапазона."""
    if not 0 <= start < end <= len(record.times):
        raise ValueError("Window must lie within the input panel")
    spec = record.spec.to_dict()
    spec.update(start=str(record.times[start]),
                stop=str(record.times[end - 1] + pd.Timedelta(spec["frequency"])))
    return PanelSpec(spec)


def window_inputs(inputs, start, end):
    """Вырезать полуоткрытое окно, сохранив маски и привязку выборки.

    Parameters
    ----------
    inputs : MaskedInputs
        Видимые входы полной панели.
    start, end : int
        Границы непустого окна в пределах панели.

    Returns
    -------
    MaskedInputs
        Входы окна с прежним началом отсчёта сетки.
    """
    return MaskedInputs(_window_spec(inputs, start, end), inputs.times[start:end], inputs.values[start:end],
                        inputs.row_present[start:end], inputs.original_observed[start:end],
                        inputs.artificial_mask[start:end], inputs.mit_mask[start:end],
                        inputs.protocol_id, inputs.population_id, inputs.population_kind)


def hide_inputs(inputs, *, artificial_mask=None, mit_mask=None):
    """Дополнительно скрыть исходные наблюдения без передачи их значений.

    Parameters
    ----------
    inputs : MaskedInputs
        Видимые входы до дополнительного скрытия.
    artificial_mask, mit_mask : ndarray of bool, optional
        Дополнительные маски формы ``inputs.values``. Маска ``mit_mask``
        используется только для обучения.

    Returns
    -------
    MaskedInputs
        Входы с объединёнными масками скрытия.
    """
    artificial = inputs.artificial_mask.copy()
    mit = inputs.mit_mask.copy()
    if artificial_mask is not None:
        artificial |= artificial_mask
    if mit_mask is not None:
        mit |= mit_mask
    return MaskedInputs(inputs.spec, inputs.times,
                        np.where(artificial | mit, np.nan, inputs.values),
                        inputs.row_present, inputs.original_observed, artificial, mit,
                        inputs.protocol_id, inputs.population_id, inputs.population_kind)


def evaluation_inputs(panel, row, regime, *, protocol_id):
    """Разделить окно банка на видимые входы и ответы целевого блока.

    Parameters
    ----------
    panel : NetworkPanel
        Исходная панель четырёх постов.
    row : object
        Блок банка с постом, полуоткрытыми границами окна и пропуска,
        геометрией и назначением выборки.
    regime : {'actual', 'none'}
        Сохранить наблюдения соседей либо скрыть их целиком в окне.
    protocol_id : str
        Идентификатор протокола.

    Returns
    -------
    inputs : MaskedInputs
        Полное скрытие блока и выбранного контекста.
    truth : EvaluationTruth
        Ответы только целевого блока, без дополнительного контекста.
    start, end : int
        Локальные полуоткрытые границы блока.
    """
    w, stop = int(row.window_start), int(row.window_end)
    hidden = np.zeros(panel.values[w:stop].shape, dtype=bool)
    inputs = MaskedInputs(_window_spec(panel, w, stop), panel.times[w:stop], panel.values[w:stop],
                          panel.row_present[w:stop], panel.original_observed[w:stop],
                          hidden, hidden, protocol_id, row.block_id, row.split)
    start, end = int(row.start) - w, int(row.end) - w
    target = int(row.site_id) if hasattr(row, "site_id") else SITES.index(row.site)
    gap = np.zeros(inputs.values.shape, dtype=bool)
    gap[start:end, target, 0] = True
    if not np.isfinite(inputs.values[gap]).all():
        raise ValueError("Evaluation truth must be originally observed")
    hidden = gap.copy()
    if row.geometry == "leading":
        hidden[:start, target, 0] = inputs.visible[:start, target, 0]
    elif row.geometry == "trailing":
        hidden[end:, target, 0] = inputs.visible[end:, target, 0]
    elif row.geometry != "internal":
        raise ValueError("Unknown evaluation geometry")
    if regime == "none":
        for j in range(len(SITES)):
            if j != target:
                hidden[:, j, 0] = inputs.visible[:, j, 0]
    elif regime != "actual":
        raise ValueError("Unknown neighbor regime")
    truth = EvaluationTruth(inputs.spec, inputs.times, np.where(gap, inputs.values, np.nan),
                            gap, inputs.protocol_id, inputs.population_id, inputs.population_kind)
    return hide_inputs(inputs, artificial_mask=hidden), truth, start, end
