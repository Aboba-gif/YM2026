"""Потоки парных ошибок, выбор наблюдений и оценка ковариации."""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, cholesky, solve_triangular
from adrkit.inverse.misfit import covariance_metric
from adrkit.spaces import ArraySpace
from experiments.source_comparison.calibration import array_hash, covariance_guards, fit_covariance, noise_covariance
from .config import DENSE_TIMES, PRIMARY_TICKS, PRIMARY_ROWS, PANEL_CODES


def standard_panel(stream, replicate, panel, index=0):
    """Сформировать воспроизводимую стандартную нормальную панель.

    Parameters
    ----------
    stream : dict
        Целочисленные seed и version случайного потока.
    replicate : int
        Номер основной реализации, начиная с 1.
    panel : str
        Зарегистрированное назначение панели: calibration, fit, selection,
        test или mask.
    index : int, optional
        Неотрицательный номер панели данного назначения.

    Returns
    -------
    values : ndarray, shape (4, 72)
        Безразмерные стандартные нормальные значения по четырём постам и 72
        временам.
    provenance : dict
        Начальное состояние PCG64, назначение и хеш панели.
    """
    if panel not in PANEL_CODES or type(replicate) is not int or replicate < 1:
        raise ValueError("Registered panel and new main replicate>=1 required")
    if type(index) is not int or index < 0:
        raise ValueError("Nonnegative panel index required")
    seed = [int(stream["seed"]), int(stream["version"]), replicate,
            PANEL_CODES[panel], index]
    z = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))).standard_normal((4,72))
    return z, dict(seed=seed, panel=panel, generator="PCG64",
                   standard_normal_sha256=array_hash(z))


def draw_residual(stream, replicate, panel, covariance, index=0):
    """Преобразовать нормальную панель в ошибки концентрации.

    Parameters
    ----------
    stream : dict
        Настройки seed и version случайного потока.
    replicate : int
        Номер основной реализации, начиная с 1.
    panel : str
        Зарегистрированное назначение панели.
    covariance : array_like, shape (288, 288)
        Положительно определённая ковариация ошибок в (мкг/м³)².
    index : int, optional
        Неотрицательный номер панели данного назначения.

    Returns
    -------
    values : ndarray, shape (288,)
        Ошибки в мкг/м³; порядок: пост, затем время.
    provenance : dict
        Описание нормального потока и хеши ковариации и ошибок.
    """

    z, provenance = standard_panel(stream, replicate, panel, index)
    values = np.linalg.cholesky(covariance) @ z.ravel()
    return values, dict(provenance, covariance_sha256=array_hash(covariance),
                        residual_sha256=array_hash(values))


def selected_rows(condition, stream, replicate):
    """Выбрать доступные строки полного календаря наблюдений.

    При маске пропусков два удалённых времени общие для всех доступных постов.

    Parameters
    ----------
    condition : dict
        Календарь primary или dense, доступность постов и маска пропусков.
    stream : dict
        Настройки seed и version случайного потока маски.
    replicate : int
        Номер реализации маски пропусков.

    Returns
    -------
    rows : ndarray of int, shape (n_selected,)
        Индексы полного набора из 288 строк в порядке постов, затем времён.
    design : dict
        Выбранные строки, число наблюдений и удалённые времена основного
        календаря.
    """
    dense = condition.get("fit_regime", "primary") == "dense"
    ticks = np.arange(72) if dense else PRIMARY_TICKS.copy()
    mask = condition["mask"]
    removed = []
    if mask != "none":
        if dense or mask not in ("random2", "block2"):
            raise ValueError("Two-of-nine masks require primary calendar")
        seed = [stream["seed"], stream["version"], replicate, PANEL_CODES["mask"]]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        removed = (sorted(rng.choice(9, 2, replace=False).tolist()) if mask == "random2"
                   else (lambda start: [start, start+1])(int(rng.integers(0,8))))
        ticks = np.delete(ticks, removed)
    stations = {"full": range(4), "onlyKrAZ": [3], "withoutKrAZ": range(3)}
    if condition["availability"] not in stations:
        raise ValueError("Unknown availability")
    rows = np.concatenate([s*72+ticks for s in stations[condition["availability"]]])
    return rows, dict(removed_primary_tick_indices=removed, rows=rows.tolist(),
                      retained_count=len(rows), mask=mask)


def restricted_metric(covariance, rows):
    """Построить метрику ошибок выбранных наблюдений.

    Parameters
    ----------
    covariance : array_like, shape (n, n)
        Ковариация полного набора наблюдений в (мкг/м³)².
    rows : array_like of int, shape (n_selected,)
        Различные допустимые индексы в нужном порядке.

    Returns
    -------
    metric : CovarianceMetric
        Метрика подматрицы выбранных строк и столбцов; единица концентрации
        — мкг/м³.

    Notes
    -----
    Ограничивается ковариация, затем строится её обратная метрика.
    Ограничение уже обратной полной матрицы даёт другую величину.
    """
    indices = np.asarray(rows)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("Nonempty integer row restriction required")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(covariance)):
        raise ValueError("Unique valid rows required")
    layout = ArraySpace((len(indices),), axes=("observation",),
        coordinates=(tuple(f"dense_row_{i}" for i in indices),), units="ug/m^3")
    return covariance_metric(np.asarray(covariance)[np.ix_(indices, indices)], layout=layout)


def observation_diagnostics(white, l2_gram):
    """Вычислить спектр и численные ранги матрицы чувствительности.

    Parameters
    ----------
    white : ndarray, shape (n_observations, n_coefficients)
        Якобиан прогноза после отбеливания в метрике ошибок наблюдений.
    l2_gram : ndarray, shape (n_coefficients, n_coefficients)
        Положительно определённая матрица штрафа L² в тех же координатах.

    Returns
    -------
    diagnostics : dict
        Сингулярные значения и ранги в ортонормированных координатах L²
        (``mass_normalized``) и в исходном базисе коэффициентов
        (``raw_coefficient_basis_dependent``). Для рангов указаны пороги.
    """
    lower=cholesky(l2_gram,lower=True)
    normalized=solve_triangular(lower,np.asarray(white).T,lower=True).T
    values=np.linalg.svd(normalized,compute_uv=False)
    maximum=float(values.max(initial=0.))
    machine_threshold=float(max(normalized.shape)*np.finfo(float).eps*maximum)
    protocol_threshold=1e-10*maximum
    raw=np.linalg.svd(white,compute_uv=False)
    raw_threshold=float(max(white.shape)*np.finfo(float).eps*raw.max(initial=0.))
    return dict(normalization="white J times L^-T, where Qref^2 M=L L^T; L2-orthonormal coordinates",
        l2_gram_sha256=array_hash(l2_gram),coefficient_count=white.shape[1],row_count=white.shape[0],
        structural_nullity_lower_bound=max(0,white.shape[1]-white.shape[0]),
        mass_normalized=dict(singular_values=values.tolist(),
            machine_rank=int(np.count_nonzero(values>machine_threshold)),machine_threshold=machine_threshold,
            protocol_relative_threshold=1e-10,protocol_absolute_threshold=protocol_threshold,
            protocol_rank=int(np.count_nonzero(values>protocol_threshold)),
            protocol_nullity=white.shape[1]-int(np.count_nonzero(values>protocol_threshold))),
        raw_coefficient_basis_dependent=dict(singular_values=raw.tolist(),
            machine_rank=int(np.count_nonzero(raw>raw_threshold)),machine_threshold=raw_threshold,
            use="diagnostic only, not comparable across coefficient bases"))


def calibrate_dense(spec, replicate, noise_id, family):
    """Построить ковариацию полного календаря наблюдений.

    Parameters
    ----------
    spec : dict
        Модели шума, настройки потока и калибровки.
    replicate : int
        Номер основной реализации, начиная с 1.
    noise_id : str
        Ключ генерирующей модели шума в разделе noise.
    family : {'W01', 'W02', 'W03', 'Woracle'}
        Общая дисперсия, дисперсии постов, экспоненциальная корреляция или
        заданная истинная ковариация.

    Returns
    -------
    covariance : ndarray, shape (288, 288)
        Ковариация четырёх постов и 72 времён в (мкг/м³)².
    provenance : dict
        Параметры модели, панелей и численной проверки.

    Notes
    -----
    Для W01–W03 параметры оцениваются по 32 независимым панелям
    формы (4, 9). На 72 времени переносится та же параметрическая модель;
    короткие лаги отдельно не калибруются. Woracle использует заданную
    генерирующую ковариацию без оценки параметров.
    """
    true_cov = noise_covariance(DENSE_TIMES, spec["noise"][noise_id])
    panels, records = [], []
    for index in range(32):
        values, record = draw_residual(spec["stream"], replicate, "calibration", true_cov, index)
        panels.append(values.reshape(4,72)[:,PRIMARY_TICKS])
        records.append(record)
    residuals = np.stack(panels)
    if family == "Woracle":
        return true_cov, dict(family=family, status="oracle", scientific_role="inaccessible true covariance diagnostic",
            covariance_sha256=array_hash(true_cov), calibration_inputs_used=False)
    fitted = fit_covariance(residuals, DENSE_TIMES[PRIMARY_TICKS], family=family,
                             specification=spec["calibration"])
    parent = fitted.provenance.to_dict()
    lag = np.abs(DENSE_TIMES[:,None]-DENSE_TIMES[None,:])
    blocks = [v*(np.eye(72) if family != "W03" else np.exp(-lag/parent["ell_hours"][i]))
              for i,v in enumerate(parent["station_variances"])]
    dense = block_diag(*blocks)
    if not np.allclose(dense[np.ix_(PRIMARY_ROWS,PRIMARY_ROWS)], fitted.covariance,
                       rtol=2e-12, atol=2e-14):
        raise ValueError("Dense parametric lift differs from calibrated primary covariance")
    guards = covariance_guards(dense, spec["calibration"]["guards"])
    return dense, dict(family=family, status="accepted", parent_coarse_calibration=parent,
        dense_covariance_sha256=array_hash(dense), dense_guards=guards,
        panel_records=records, calibration_shape=list(residuals.shape),
        dense_short_lags="parametric extrapolation, not independently calibrated")
