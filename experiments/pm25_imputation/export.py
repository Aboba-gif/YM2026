"""Выбор методов на искусственных пропусках и экспорт PM₂.₅ после переобучения."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
from timeseries.metrics import MetricSet, metric_from_function

from .models import reorder
from .imputation import pm25_imputer, prediction_results
from .panel import evaluation_inputs, network_panel, visible_inputs, window_inputs
from .smoothing import smooth_pm25, smoothing_diagnostics
from .data import SITES, load_network, pm25_matrix, file_sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).resolve().parent / 'resources/protocol.json'
METHODS = ['linear', 'common_hgb', 'network_hgb', 'climatology']
_BLOCK_METRICS = MetricSet((
    metric_from_function('squared_sum', lambda y, p: (e := p - y) @ e,
                         unit='channel_squared'),
    metric_from_function('absolute_sum', lambda y, p: np.abs(p - y).sum(),
                         unit='channel'),
    metric_from_function('error_sum', lambda y, p: (p - y).sum(), unit='channel'),
    metric_from_function('rmse', lambda y, p: np.sqrt(np.mean((e := p - y) * e)),
                         unit='channel'),
    metric_from_function('mae', lambda y, p: np.mean(np.abs(p - y)), unit='channel'),
    metric_from_function('bias', lambda y, p: (p - y).mean(), unit='channel'),
))


def dump(path, value):
    """Записать значение в UTF-8 JSON, создав родительский каталог.

    Parameters
    ----------
    path : str or Path
        Путь выходного файла; существующий файл перезаписывается.
    value : JSON-serializable object
        Значение для записи; NaN и бесконечные числа не допускаются.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def runs(mask):
    """Найти максимальные непрерывные участки истинных значений маски.

    Parameters
    ----------
    mask : array_like, shape (n,)
        Маска, приводимая к булевым значениям.

    Returns
    -------
    list of tuple of int
        Пары начального и конечного индексов; правый индекс не включается.
    """

    change = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(change == 1).tolist(), np.flatnonzero(change == -1).tolist()))


def make_bank(index, values, cfg):
    """Выбрать окна искусственных пропусков по исходной доступности.

    Parameters
    ----------
    index : DatetimeIndex
        Индекс `n` отсчётов; для моделей и экспорта используется регулярная
        сетка с шагом 20 минут.
    values : ndarray, shape (n, 4)
        Исходные концентрации PM₂.₅ в единицах исходного экспорта в порядке `SITES`; NaN обозначает
        пропуск.
    cfg : dict
        Конфигурация протокола заполнения PM₂.₅.

    Returns
    -------
    bank : DataFrame
        Блоки каждого поста, части данных, длины и геометрии с индексами окон
        и целевых блоков.
    counts : DataFrame
        Числа кандидатных, допустимых, запрошенных и выбранных окон.

    Notes
    -----
    Отбор требует исходно наблюдавшихся значений целевого блока и его
    концов. Сезонные пулы перемешиваются с заданным `seed`; концентрации
    и ошибки моделей не задают порогов отбора.
    """
    rows, counts = [], []
    for site_id, site in enumerate(SITES):
        for split_id, (split, dates) in enumerate(cfg['splits'].items()):
            positions = np.flatnonzero((index >= dates[0]) & (index < dates[1]))
            for geo_id, geometry in enumerate(['internal', 'leading', 'trailing']):
                lengths = cfg['bank']['internal_lengths' if geometry == 'internal' else 'boundary_lengths']
                requested = cfg['bank']['requested' if geometry == 'internal' else 'boundary_requested'][split]
                for length in lengths:
                    width = max(288, length + 2 * cfg['bank']['margin'])
                    candidates = list(range(int(positions[0]), int(positions[-1]) - width + 2, width))
                    eligible = [w for w in candidates if np.isfinite(values[w + (width-length)//2 - 1:w + (width-length)//2 + length + 1, site_id]).all()]
                    rng = np.random.default_rng(cfg['seed'] + 100000*site_id + 10000*split_id + 1000*geo_id + length)
                    pools = [list(rng.permutation([w for w in eligible if (index[w].month % 12)//3 == season])) for season in range(4)]
                    chosen = []
                    while len(chosen) < requested and any(pools):
                        for pool in pools:
                            if pool and len(chosen) < requested:
                                chosen.append(int(pool.pop()))
                    counts.append(dict(site=site, split=split, geometry=geometry, length=length, candidate_windows=len(candidates), eligible_windows=len(eligible), requested=requested, selected=len(chosen)))
                    for k, w in enumerate(sorted(chosen)):
                        a = w + (width-length)//2
                        rows.append(dict(block_id=f'{site}_{split}_{geometry}_L{length}_{k:02d}', site=site, site_id=site_id, split=split, geometry=geometry, length=length, window_start=w, window_end=w+width, start=a, end=a+length, start_time=str(index[a])))
    return pd.DataFrame(rows), pd.DataFrame(counts)




def error_sums(truth, pred):
    """Вычислить суммы и средние ошибок одного блока.

    Parameters
    ----------
    truth, pred : ndarray of float or int, shape (n,)
        Непустые конечные контрольные значения и прогнозы в единицах
        исходного числового экспорта. Вычисления выполняются в float64.

    Returns
    -------
    dict
        Число `n`, суммы квадратов, модулей и знаковых ошибок, RMSE, MAE и
        среднюю ошибку. Ошибка — разность прогноза и контрольного значения (``pred - truth``). ``squared_sum``
        имеет квадрат единицы входа; остальные ошибки — единицу входа.

    Raises
    ------
    ValueError
        Пустой блок, неверная форма или численный тип, неконечные значения
        либо неконечный результат метрики.
    TypeError
        Прогноз имеет строковый или объектный тип.
    """

    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError('Incomplete candidate prediction')
    if not len(truth):
        raise ValueError('Evaluation block must be nonempty')
    return dict(n=len(truth), **_BLOCK_METRICS.score(truth, pred))


def evaluate(panel, bank, models, cfg):
    """Оценить доступные методы на скрытых блоках проверочного банка.

    Parameters
    ----------
    panel : NetworkPanel
        Исходные значения четырёх постов, сетка по 20 минут и маски
        присутствия. Физическая единица экспорта не подтверждена.
    bank : DataFrame
        Блоки, созданные `make_bank`, с постом, частью данных, геометрией и
        границами.
    models : dict of str to PM25Fit
        Обученная модель каждого поста из `SITES`.
    cfg : dict
        Конфигурация протокола заполнения PM₂.₅.

    Returns
    -------
    DataFrame
        Ошибка каждого доступного метода для каждого блока и режима соседей;
        неприменимые методы не создают строк.
    """

    records = []
    for site in SITES:
        for row in bank[bank.site.eq(site)].itertuples():
            for regime in cfg['bank']['neighbor_regimes']:
                inputs, truth, a, b = evaluation_inputs(panel, row, regime, protocol_id=cfg['protocol_id'])
                results = prediction_results(inputs, models[site], a, b)
                for method, result in results.items():
                    result.require_binding(truth)
                    available = result.availability[truth.target_mask]
                    if available.all():
                        target = truth.values[truth.target_mask]
                        pred = result.prediction[truth.target_mask]
                        records.append(dict(block_id=row.block_id, site=site, split=row.split, geometry=row.geometry, length=row.length, regime=regime, method=method, **error_sums(target, pred)))
                    elif available.any():
                        raise ValueError('Incomplete candidate prediction')
    return pd.DataFrame(records)


def aggregate(frame, keys):
    """Объединить ошибки блоков с равным весом каждого вхождения отсчёта.

    Parameters
    ----------
    frame : DataFrame
        Строки с `n`, `squared_sum`, `absolute_sum` и `error_sum` из
        `error_sums`.
    keys : sequence of str
        Колонки группировки.

    Returns
    -------
    DataFrame
        Число блоков и отсчётов, RMSE, MAE и средняя ошибка (`bias`) по всем
        вхождениям отсчётов каждой группы; ошибки в единицах исходного экспорта.
    """

    rows = []
    for labels, group in frame.groupby(keys, sort=True, dropna=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        n = int(group.n.sum())
        rows.append({**dict(zip(keys, labels)), 'blocks': len(group), 'n': n,
                     'rmse': float(np.sqrt(group.squared_sum.sum()/n)),
                     'mae': float(group.absolute_sum.sum()/n), 'bias': float(group.error_sum.sum()/n)})
    return pd.DataFrame(rows)


def select_policy(selection, cfg):
    """Выбрать методы по среднеквадратичной ошибке на выборке `selection`.

    Parameters
    ----------
    selection : DataFrame
        Непустые результаты `evaluate`, содержащие только строки части
        `selection`.
    cfg : dict
        Конфигурация протокола заполнения PM₂.₅.

    Returns
    -------
    dict
        Политика для внутренних длин и резервного метода каждого поста и
        режима соседей с данными о доступности блоков выборки `selection`.

    Notes
    -----
    Метод допускается при достаточном числе блоков. При равной ошибке
    выбирается первый кандидат в конфигурации; при нехватке данных
    используется `climatology`. Данные выборки `test` не участвуют в выборе.
    """

    if selection.empty or not selection.split.eq('selection').all():
        raise ValueError('Policy selection accepts only selection rows')
    internal, fallback = [], []
    minimum = cfg['selection']['minimum_blocks']
    def choose(part, candidates):
        scores = []
        for order, name in enumerate(candidates):
            g = part[part.method.eq(name)]
            if len(g) >= minimum:
                scores.append((float(g.squared_sum.sum()/g.n.sum()), order, name, len(g)))
        if not scores:
            return dict(method='climatology', selection_blocks=0, selection_rmse=None, selection_available=False)
        mse, _, name, n = min(scores)
        return dict(method=name, selection_blocks=n, selection_rmse=float(np.sqrt(mse)), selection_available=True)
    for site in SITES:
        for regime in cfg['bank']['neighbor_regimes']:
            part = selection[selection.site.eq(site) & selection.regime.eq(regime)]
            for length in cfg['bank']['internal_lengths']:
                g = part[part.geometry.eq('internal') & part.length.eq(length)]
                internal.append(dict(site=site, regime=regime, length=length, **choose(g, cfg['selection']['internal_candidates'])))
            g = part[part.geometry.ne('internal') & part.length.eq(720)]
            population = 'leading_and_trailing_720'
            if g[g.method.eq('climatology')].shape[0] < minimum:
                g = part[part.geometry.eq('internal') & part.length.eq(720)]
                population = 'internal_720_due_to_boundary_shortfall'
            fallback.append(dict(site=site, regime=regime, population=population, **choose(g, cfg['selection']['fallback_candidates'])))
    return dict(protocol_id=cfg['protocol_id'], internal=internal, fallback=fallback,
                selection_period=cfg['splits']['selection'], test_used_for_selection=False,
                intervals='none; no natural-gap coverage guarantee', final_refit='all original observed2019-2022 labels')


def policy_choice(policy, site, length, regime, geometry='internal'):
    """Найти метод для длины и геометрии естественного пропуска.

    Parameters
    ----------
    policy : dict
        Результат `select_policy`.
    site : str
        Пост из `SITES`.
    length : int
        Длина пропуска в отсчётах по 20 минут.
    regime : {'actual', 'none'}
        Режим доступности соседей.
    geometry : {'internal', 'leading', 'trailing'}, optional
        Расположение пропуска; по умолчанию `internal`.

    Returns
    -------
    dict
        Запись политики ближайшей допустимой длины не меньше `length` для
        `internal` либо резервная запись. Возвращается объект из `policy`.
    """

    if geometry == 'internal':
        available = [r for r in policy['internal'] if r['site']==site and r['regime']==regime and r['length']>=length]
        if available:
            return min(available, key=lambda r:r['length'])
    return next(r for r in policy['fallback'] if r['site']==site and r['regime']==regime)


def selected_metrics(blocks, policy):
    """Оставить ошибку метода, выбранного политикой для каждого блока.

    Parameters
    ----------
    blocks : DataFrame
        Результаты `evaluate` с `block_id` и `regime`.
    policy : dict
        Политика `select_policy`.

    Returns
    -------
    DataFrame
        Ровно одна строка выбранного метода на каждый блок и режим.

    Raises
    ------
    ValueError
        Выбранный метод отсутствует или представлен несколькими строками.
    """

    selected = []
    for _, g in blocks.groupby(['block_id', 'regime'], sort=True):
        row = g.iloc[0]
        choice = policy_choice(policy, row.site, int(row.length), row.regime, row.geometry)
        chosen = g[g.method.eq(choice['method'])]
        if len(chosen) != 1:
            raise ValueError('Selected method absent from evaluation bank')
        selected.append(chosen.iloc[0].to_dict())
    return pd.DataFrame(selected)


def support_status(start, end, observed, choice, prediction_available=True):
    """Определить статус применимости проверки к естественному пропуску.

    Parameters
    ----------
    start, end : int
        Полуоткрытые границы пропуска в полном ряду.
    observed : ndarray of bool, shape (n,)
        Маска исходной доступности; здесь её длина задаёт конец ряда.
    choice : dict
        Запись политики с `selection_available`.
    prediction_available : bool, optional
        Доступность первоначально выбранного метода, по умолчанию True.

    Returns
    -------
    str
        Статус граничной, длинной или неподдержанной экстраполяции либо
        `artificial_mask_evaluated`. Статус описывает применимость
        искусственной проверки, не точность естественного пропуска.
    """

    if start == 0:
        return 'leading_extrapolation'
    if end == len(observed):
        return 'trailing_extrapolation'
    if end-start > 720:
        return 'long_gap_extrapolation'
    if not choice['selection_available'] or not prediction_available:
        return 'method_support_extrapolation'
    return 'artificial_mask_evaluated'


def build_export(panel, models, policy, cfg):
    """Собрать заполненные и сглаженные ряды с маркировкой происхождения.

    Parameters
    ----------
    panel : NetworkPanel
        Исходные значения четырёх постов, сетка по 20 минут и маски
        присутствия. Физическая единица экспорта не подтверждена.
    models : dict of str to PM25Fit
        Обученная модель каждого поста из `SITES`.
    policy : dict
        Выбранная политика внутренних и резервных методов.
    cfg : dict
        Конфигурация протокола заполнения PM₂.₅.

    Returns
    -------
    output : DataFrame
        Временной индекс и исходные, заполненные, сглаженные значения
        в единицах исходного экспорта с масками, методами и статусами постов.
    ledger : DataFrame
        Запись каждого естественного пропуска с длиной, методом, режимом
        соседей и поддержкой искусственной проверки.
    annual : DataFrame
        Годовые числа наблюдений, заполненных значений и статусов по постам.
    smoothing : DataFrame
        Диагностика каждой настройки сглаживания по постам.

    Notes
    -----
    Исходные наблюдения точно сохраняются в заполненном ряду.
    Сглаженный ряд может их изменить. `adr_observation_weight` равен единице
    только для исходных наблюдений; заполнение не создаёт новых наблюдений.
    """

    index = pd.DatetimeIndex(panel.times)
    values = panel.values[:, :, 0]
    inputs = visible_inputs(panel, protocol_id=cfg['protocol_id'],
                            population_id='natural-export', population_kind='export')
    output = pd.DataFrame({'timestamp': index.astype(str)})
    ledger, annual, smoothing = [], [], []
    for j, site in enumerate(SITES):
        raw = values[:, j]
        observed = np.isfinite(raw)
        filled = raw.copy()
        source = np.full(len(raw), 'observed', dtype=object)
        method = np.full(len(raw), 'observed', dtype=object)
        length_col = np.zeros(len(raw), dtype=int)
        target = reorder(values, j)
        for start, end in runs(~observed):
            bounds = models[site].models.common_window_bounds(len(target), start, end)
            if bounds is None:
                neighbor_context = target[start:end, 1:]
            else:
                window_start, window_end = bounds
                neighbor_context = target[window_start:window_end, 1:]
            regime = 'none' if not np.isfinite(neighbor_context).any() else 'actual'
            geometry = 'leading' if start==0 else 'trailing' if end==len(raw) else 'internal'
            choice = policy_choice(policy, site, end-start, regime, geometry)
            if bounds is not None:
                window_start, window_end = bounds
            elif len(raw) < models[site].models.config['common_window_steps']:
                window_start, window_end = 0, len(raw)
            else:
                window_start, window_end = max(0, start - 1), min(len(raw), end + 1)
            window = window_inputs(inputs, window_start, window_end)
            local_start, local_end = start - window_start, end - window_start
            results = prediction_results(window, models[site], local_start, local_end)
            predictions = {}
            for name, result in results.items():
                result.require_binding(window)
                available = result.availability[local_start:local_end, j, 0]
                if available.all():
                    predictions[name] = result.prediction[local_start:local_end, j, 0]
                elif not available.any():
                    predictions[name] = None
                else:
                    raise ValueError('Incomplete candidate prediction')
            selected = choice['method']
            applicable = predictions[selected] is not None
            if not applicable:
                choice = policy_choice(policy, site, end-start, regime, 'leading')
                selected = choice['method']
            status = support_status(start, end, observed, choice, applicable)
            filled[start:end] = predictions[selected]
            source[start:end], method[start:end], length_col[start:end] = status, selected, end-start
            ledger.append(dict(site=site, start=str(index[start]), end=str(index[end-1]), start_index=start, end_index_exclusive=end,
                               length=end-start, hours=(end-start)/3, status=status, method=selected, neighbor_regime=regime,
                               observed_neighbor_fraction=float(np.isfinite(target[start:end, 1:]).mean()),
                               observed_neighbor_context_fraction=float(np.isfinite(neighbor_context).mean()),
                               selection_reference_length=choice.get('length'), selection_reference_rmse=choice['selection_rmse'],
                               error_reference_kind='artificial-selection pooled RMSE; not per-point uncertainty'))
        if not np.isfinite(filled).all() or (filled<0).any() or not np.array_equal(raw[observed], filled[observed]):
            raise ValueError('Final completion violated finiteness/positivity/original preservation')
        weights = np.where(observed, cfg['smoothing']['weights']['observed'],
                           np.where(source=='artificial_mask_evaluated', cfg['smoothing']['weights']['artificial_mask_evaluated'], cfg['smoothing']['weights']['extrapolation']))
        final_smooth = None
        for lam in cfg['smoothing']['sensitivity_lambdas']:
            z = smooth_pm25(filled, weights=weights, lam=lam)
            if lam == cfg['smoothing']['lambda']:
                final_smooth = z
            smoothing.append(dict(site=site, lam=lam, **smoothing_diagnostics(filled, z, weights=weights, lam=lam),
                                  observed_rmse_change=float(np.sqrt(np.mean((z[observed]-raw[observed])**2))),
                                  max_absolute_change=float(np.max(np.abs(z-filled))),
                                  maximum_before=float(filled.max()), maximum_after=float(z.max())))
        if final_smooth is None:
            raise ValueError('Final smoothing parameter absent from sensitivity grid')
        for name, array in [('pm25_observed', raw), ('pm25_filled', filled), ('pm25_smoothed', final_smooth),
                            ('is_imputed', (~observed).astype(int)), ('status', source), ('method', method),
                            ('gap_steps', length_col), ('adr_observation_weight', observed.astype(int))]:
            output[f'{site}_{name}'] = array
        for year in np.unique(index.year):
            use = index.year==year
            row = dict(site=site, year=int(year), rows=int(use.sum()), observed=int(observed[use].sum()), imputed=int((~observed[use]).sum()), remaining_filled_nan=int(np.isnan(filled[use]).sum()))
            for status in ['artificial_mask_evaluated', 'leading_extrapolation', 'trailing_extrapolation', 'long_gap_extrapolation', 'method_support_extrapolation']:
                row[status] = int(np.sum(source[use]==status))
            annual.append(row)
    return output, pd.DataFrame(ledger), pd.DataFrame(annual), pd.DataFrame(smoothing)


def verify_export(csv_path, index, values):
    """Проверить записанный CSV относительно исходных значений и их происхождения.

    Parameters
    ----------
    csv_path : str or Path
        Записанный канонический CSV экспорта.
    index : DatetimeIndex
        Индекс `n` отсчётов; для моделей и экспорта используется регулярная
        сетка с шагом 20 минут.
    values : ndarray, shape (n, 4)
        Исходные концентрации PM₂.₅ в единицах исходного экспорта в порядке `SITES`; NaN обозначает
        пропуск.

    Returns
    -------
    dict
        Результат проверки временной сетки, схемы, исходных значений, масок,
        неотрицательности и статусов с SHA-256 файла.

    Notes
    -----
    Проверка сериализации и маркировки не оценивает ошибку заполнения
    естественных пропусков.
    """
    frame = pd.read_csv(csv_path, float_precision='round_trip')
    if not pd.DatetimeIndex(pd.to_datetime(frame.timestamp)).equals(index):
        raise ValueError('Export clock, ordering, length or frequency changed')
    if len(frame.columns) != 33 or not frame.columns.is_unique:
        raise ValueError('Unexpected canonical CSV schema')
    records = []
    for j, site in enumerate(SITES):
        raw = values[:, j]
        observed = np.isfinite(raw)
        exported_raw = frame[f'{site}_pm25_observed'].to_numpy()
        filled = frame[f'{site}_pm25_filled'].to_numpy()
        smoothed = frame[f'{site}_pm25_smoothed'].to_numpy()
        if not np.array_equal(raw, exported_raw, equal_nan=True):
            raise ValueError('Serialized original values or original masks changed')
        if not np.array_equal(raw[observed], filled[observed]):
            raise ValueError('An original observation was overwritten in filled series')
        if not np.isfinite(filled).all() or not np.isfinite(smoothed).all() or min(filled.min(), smoothed.min())<0:
            raise ValueError('Nonfinite or negative completion')
        if not np.array_equal(frame[f'{site}_is_imputed'].to_numpy(), (~observed).astype(int)) or not np.array_equal(frame[f'{site}_adr_observation_weight'].to_numpy(), observed.astype(int)):
            raise ValueError('Invalid raw/derived provenance mask')
        status = frame[f'{site}_status'].to_numpy()
        method = frame[f'{site}_method'].to_numpy()
        if not np.all(status[observed]=='observed') or not np.all(method[observed]=='observed') or not np.all(frame.loc[observed, f'{site}_gap_steps']==0):
            raise ValueError('Observed provenance corrupted')
        allowed = {'artificial_mask_evaluated', 'leading_extrapolation', 'trailing_extrapolation', 'long_gap_extrapolation', 'method_support_extrapolation'}
        if not set(status[~observed]) <= allowed or not set(method[~observed]) <= set(METHODS):
            raise ValueError('Unknown imputation status/method')
        for a,b in runs(~observed):
            if len(set(status[a:b])) != 1 or len(set(method[a:b])) != 1:
                raise ValueError('Method/status must be constant within a natural run')
            if not np.all(frame.loc[a:b-1, f'{site}_gap_steps']==b-a):
                raise ValueError('Gap length provenance corrupted')
            required = 'leading_extrapolation' if a==0 else 'trailing_extrapolation' if b==len(raw) else 'long_gap_extrapolation' if b-a>720 else None
            if required and not np.all(status[a:b]==required):
                raise ValueError('Unvalidated boundary/long interval mislabeled')
            if required is None and status[a] not in {'artificial_mask_evaluated','method_support_extrapolation'}:
                raise ValueError('Internal short interval assigned a false boundary/long status')
        records.append(dict(site=site, rows=len(raw), observed=int(observed.sum()), imputed=int((~observed).sum()), filled_nan=0, original_exact=True, nonnegative=True))
    return dict(passed=True, rows=len(frame), columns=len(frame.columns), stations=records, csv_sha256=file_sha(csv_path), bytes=Path(csv_path).stat().st_size,
                assurance='serialization, counts, provenance, positivity; no natural-gap accuracy claim')


def _validate_periods(index, cfg):
    """Проверить периоды обучения, выбора и проверки на временной сетке.

    Parameters
    ----------
    index : DatetimeIndex
        Регулярный временной индекс исходных данных без часового пояса.
    cfg : dict
        Протокол с ``period``, ``frequency``, ``train_end`` и ``splits``.

    Raises
    ------
    ValueError
        Периоды пусты, пересекаются, выходят за временную сетку или их
        границы не совпадают с отсчётами. Правая граница не включается.
    """
    def timestamp(value):
        if not isinstance(value, str):
            raise ValueError('Period boundaries must be date strings')
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tz is not None:
            raise ValueError('Period boundaries must be finite and timezone-naive')
        return result

    def bounds(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('A period must have a start and an exclusive end')
        return tuple(timestamp(item) for item in value)

    period_start, period_end = bounds(cfg['period'])
    step = pd.Timedelta(cfg['frequency'])
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError('Frequency must be a positive fixed duration')
    if len(index) < 2 or index.tz is not None or not np.all(index[1:] - index[:-1] == step):
        raise ValueError('Protocol frequency must match the regular raw clock')
    if period_start != index[0] or period_end != index[-1] + step:
        raise ValueError('Protocol period must match the complete raw clock')
    train_end = timestamp(cfg['train_end'])
    selection = bounds(cfg['splits']['selection'])
    test = bounds(cfg['splits']['test'])
    for start, end in ((period_start, train_end), selection, test):
        if not period_start <= start < end <= period_end:
            raise ValueError('Training, selection and test periods must be nonempty and in range')
        if (start - period_start) % step or (end - period_start) % step:
            raise ValueError('Period boundaries must align with the raw clock')
    if train_end > selection[0] or selection[1] > test[0]:
        raise ValueError('Training, selection and test periods must be ordered and disjoint')


def run(data_root, *, output_dir, results_dir, protocol_path=None, verify_only=False, contract=None,
        protected_roots=()):
    """Выбрать методы, переобучить модели и записать экспорт заполненного PM₂.₅.

    Parameters
    ----------
    data_root : str or Path
        Каталог исходных годовых CSV четырёх постов.
    output_dir : str or Path
        Каталог итогового CSV, журнала пропусков, политики и `manifest.json`.
    results_dir : str or Path
        Каталог проверочных блоков, метрик и записей обучения.
    protocol_path : str or Path or None, optional
        JSON протокола; None использует `DEFAULT_PROTOCOL`.
    verify_only : bool, optional
        Проверить существующий итоговый CSV, не обучая модели и не выполняя
        экспорт.
    contract : str or Path or None, optional
        Существующий файл, копируемый как README.md после экспорта; None не
        добавляет файл.
    protected_roots : iterable of str or Path, optional
        Дополнительные входные пути, с которыми вывод не должен пересекаться.

    Returns
    -------
    dict
        Сведения о выполненном экспорте; при `verify_only` — отчёт
        `verify_export`.
    """
    from types import SimpleNamespace
    from experiments.pm25_imputation.output import outputs
    import timeseries
    args = SimpleNamespace(data_root=Path(data_root).resolve(), output_dir=Path(output_dir).absolute(),
                           results_dir=Path(results_dir).absolute(),
                           protocol=Path(protocol_path) if protocol_path is not None else DEFAULT_PROTOCOL,
                           verify_only=verify_only, contract=Path(contract) if contract is not None else None)
    if not verify_only:
        protected = [args.data_root, args.protocol, ROOT, Path(timeseries.__file__).resolve().parent,
                     *protected_roots,*([args.contract] if args.contract is not None else [])]
        args.output_dir = outputs(args.output_dir, ('pm25_final_2019_2022.csv', 'gap_ledger.csv',
                                                    'manifest.json', 'policy.json', 'README.md'), protected=protected)
        args.results_dir = outputs(args.results_dir, ('validation_bank.csv', 'validation_eligibility.csv',
            'validation_fits.json', 'policy.json', 'block_metrics.csv', 'metrics.csv', 'selected_block_metrics.csv',
            'selected_metrics.csv', 'final_fits.json', 'annual_summary.csv', 'smoothing_sensitivity.csv',
            'verification.json', 'validation_support.json', 'export_manifest.json'), protected=protected)
    started = time.time()
    protocol_bytes = args.protocol.read_bytes()
    cfg = json.loads(protocol_bytes)
    source_paths = [Path(__file__).parent/name for name in ['export.py','models.py','smoothing.py','data.py','panel.py','features.py','imputation.py']]
    source_manifest = {str(p.relative_to(ROOT)).replace('\\','/'): file_sha(p) for p in source_paths}
    protocol_digest = hashlib.sha256(protocol_bytes).hexdigest()
    index, frames, raw_manifest = load_network(args.data_root)
    _validate_periods(index, cfg)
    panel = network_panel(index, pm25_matrix(frames), row_present=np.column_stack([
        frames[site].row_present.to_numpy() for site in SITES]))
    values = panel.values[:, :, 0]
    destination = args.output_dir/'pm25_final_2019_2022.csv'
    if args.verify_only:
        verification = verify_export(destination, index, values)
        print(json.dumps(verification, indent=2))
        return verification
    import sklearn
    from threadpoolctl import threadpool_limits
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rdir = args.results_dir
    bank, eligibility = make_bank(index, values, cfg)
    bank.to_csv(rdir/'validation_bank.csv', index=False)
    eligibility.to_csv(rdir/'validation_eligibility.csv', index=False)
    model_config = {k:v for k,v in cfg['models'].items() if k in {'hgb', 'common_window_steps', 'common_stride', 'common_lengths', 'common_max_windows'}}
    train = np.asarray(index < cfg['train_end'])
    training = visible_inputs(panel, protocol_id=cfg['protocol_id'],
                              population_id='validation-train', population_kind='train', train_mask=train)
    models = {}
    with threadpool_limits(limits=1):
        for j, site in enumerate(SITES):
            print('Обучение для выбора метода:', site, flush=True)
            models[site] = pm25_imputer(model_config, site).fit(
                training, rng=np.random.default_rng(cfg['seed']), train_mask=train)
        dump(rdir/'validation_fits.json', {s:m.models.fit_record for s,m in models.items()})
        selection = evaluate(panel, bank[bank.split.eq('selection')], models, cfg)
        policy = select_policy(selection, cfg)
        dump(rdir/'policy.json', policy)
        policy_sha = file_sha(rdir/'policy.json')
        print('Правило выбора сохранено; оценка качества на выборке test', flush=True)
        test = evaluate(panel, bank[bank.split.eq('test')], models, cfg)
        if file_sha(rdir/'policy.json') != policy_sha:
            raise ValueError('Policy changed after test evaluation')
        blocks = pd.concat([selection, test], ignore_index=True)
        blocks.to_csv(rdir/'block_metrics.csv', index=False)
        aggregate(blocks, ['site','split','geometry','length','regime','method']).to_csv(rdir/'metrics.csv',index=False)
        chosen = selected_metrics(blocks, policy)
        chosen.to_csv(rdir/'selected_block_metrics.csv',index=False)
        aggregate(chosen, ['site','split','geometry','length','regime']).to_csv(rdir/'selected_metrics.csv',index=False)
        final_train = np.ones(len(index), dtype=bool)
        final_inputs = visible_inputs(panel, protocol_id=cfg['protocol_id'],
                                      population_id='final-train', population_kind='train')
        for j, site in enumerate(SITES):
            print('Обучение для заполнения по всем исходным наблюдениям:', site, flush=True)
            models[site] = pm25_imputer(model_config, site).fit(
                final_inputs, rng=np.random.default_rng(cfg['seed']), train_mask=final_train)
        dump(rdir/'final_fits.json', {s:m.models.fit_record for s,m in models.items()})
        output, ledger, annual, smoothing = build_export(panel, models, policy, cfg)
    output.to_csv(destination, index=False, lineterminator='\n')
    ledger.to_csv(args.output_dir/'gap_ledger.csv',index=False)
    annual.to_csv(rdir/'annual_summary.csv',index=False)
    smoothing.to_csv(rdir/'smoothing_sensitivity.csv',index=False)
    verification = verify_export(destination, index, values)
    for record in raw_manifest:
        if file_sha(args.data_root/record['file']) != record['sha256']:
            raise ValueError('Original CSV changed during export')
    if file_sha(args.protocol) != protocol_digest or any(file_sha(ROOT/p) != digest for p,digest in source_manifest.items()):
        raise ValueError('Protocol or computation source changed during export')
    dump(rdir/'verification.json',verification)
    counts = []
    for (site,split), group in bank.groupby(['site','split']):
        unique = set(i for row in group.itertuples() for i in range(row.start,row.end))
        counts.append(dict(site=site,split=split,blocks=len(group),target_occurrences=int(group.length.sum()),unique_hidden_timestamps=len(unique),overlap_is_not_independence=True))
    dump(rdir/'validation_support.json',counts)
    manifest = dict(protocol_id=cfg['protocol_id'], protocol_sha256=protocol_digest, policy_sha256=policy_sha,
                    executed_sources=source_manifest, sources_unchanged_during_run=True,
                    execution_paths=dict(data_root=str(args.data_root), output_dir=str(args.output_dir),
                                         results_dir=str(args.results_dir), protocol=str(args.protocol.resolve())),
                    raw=raw_manifest, raw_unchanged=True, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                    thread_limit=1, runtime_seconds=time.time()-started,
                    artifacts=[dict(file=p.name,sha256=file_sha(p),bytes=p.stat().st_size) for p in [destination,args.output_dir/'gap_ledger.csv']],
                    interpretation=cfg['interpretation'], smoothing=cfg['smoothing'], verification=verification)
    dump(args.output_dir/'manifest.json',manifest)
    dump(rdir/'export_manifest.json',manifest)
    dump(args.output_dir/'policy.json',policy)
    contract = args.contract
    if contract is not None and contract.exists():
        shutil.copyfile(contract, args.output_dir/'README.md')
    print(json.dumps(verification,indent=2),flush=True)
    return manifest


def main(argv=None):
    """Передать аргументы общему CLI заполнения PM₂.₅.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы без имени программы; None использует ``sys.argv[1:]``.

    Returns
    -------
    int
        Код завершения CLI; при успешном выполнении 0.
    """

    from experiments.pm25_imputation.cli import main as run_cli
    return run_cli(argv)


if __name__ == '__main__':
    main()
