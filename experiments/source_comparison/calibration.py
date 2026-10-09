"""Парные потоки ошибок и независимо оценённые фиксированные ковариации."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.linalg import block_diag, solve_triangular

from adrkit.config.validation import JSONRecord
from adrkit.inverse.misfit import covariance_metric
from .arrays import finite_real_array


def array_hash(values):
    """Вычислить SHA-256 массива с учётом его формы.

    Parameters
    ----------
    values : array_like
        Числовые значения; перед хешированием преобразуются в little-endian
        float64 в порядке C.

    Returns
    -------
    digest : str
        Шестнадцатеричный хеш формы, типа и байтов массива.
    """
    array = np.asarray(values, dtype="<f8", order="C")
    header = JSONRecord({"shape": list(array.shape), "dtype": "<f8"}).sha256
    return sha256(header.encode("ascii")+array.tobytes(order="C")).hexdigest()


def _frozen(values):
    result = np.array(values, dtype=float, copy=True)
    result.flags.writeable = False
    return result


class CalibrationFailure(ValueError):
    """Ошибка оценки ковариации со статусом отказа калибровки.

    Parameters
    ----------
    reason : str
        Причина, передаваемая в сообщение ValueError.

    Attributes
    ----------
    status : str
        Код ``unidentifiable_calibration`` для отклонённой оценки.
    """
    def __init__(self, reason):
        self.status = "unidentifiable_calibration"
        super().__init__(reason)


def noise_covariance(times, specification):
    """Построить ковариацию ошибок концентрации для независимых постов.

    Parameters
    ----------
    times : array_like, shape (n_times,)
        Возрастающие моменты наблюдений в часах.
    specification : dict
        Семейство ошибок и стандартные отклонения по постам в мкг/м³.
        Для экспоненциальной корреляции также задаются её длины в часах.

    Returns
    -------
    covariance : ndarray, shape (n_stations * n_times, n_stations * n_times)
        Блочно-диагональная матрица в (мкг/м³)². Строки упорядочены
        по постам, а внутри каждого поста — по времени.
    """
    times = finite_real_array(times, "noise times")
    sd = finite_real_array(specification["station_sd"], "station SD")
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or sd.ndim != 1 or not len(sd) or np.any(sd <= 0)):
        raise ValueError("increasing times and positive station SD required")
    kind = specification["type"]
    if kind == "station_iid":
        return np.diag(np.repeat(sd**2, len(times)))
    if kind != "station_exponential":
        raise ValueError("unregistered noise family")
    ell = finite_real_array(specification["ell_hours"], "correlation lengths")
    if ell.shape != sd.shape or np.any(ell <= 0):
        raise ValueError("one positive correlation length per station required")
    lags = np.abs(times[:, None]-times[None, :])
    return block_diag(*[s*s*np.exp(-lags/l) for s, l in zip(sd, ell)])


@dataclass(frozen=True)
class ResidualPanel:
    """Ошибки концентрации одной панели и описание их генерации.

    Parameters
    ----------
    values : ndarray, shape (n_stations, n_times)
        Ошибки в мкг/м³, упорядоченные по постам и времени.
    provenance : JSONRecord
        Модель шума, начальное состояние генератора и хеши значений.
    """

    values: np.ndarray
    provenance: JSONRecord


def draw_panel(binding, *, noise_id, replicate, panel, index=0):
    """Сформировать воспроизводимую панель гауссовых ошибок концентрации.

    Parameters
    ----------
    binding : ProtocolBinding
        Протокол с моделью ошибок и настройками случайных потоков.
    noise_id : str
        Идентификатор модели ошибок в протоколе.
    replicate : int
        Номер реализации шума.
    panel : str
        Назначение панели: калибровка, подгонка, выбор или проверка.
    index : int, optional
        Номер панели данного назначения, начиная с нуля.

    Returns
    -------
    residual_panel : ResidualPanel
        Ошибки в мкг/м³ по постам и временам и описание случайного потока.
    """
    protocol = binding.document.to_dict()
    rng_spec = protocol["reproducibility"]
    count = protocol["calibration_and_splits"]["panels"].get(panel)
    if (count is None or type(index) is not int or not 0 <= index < count
            or type(replicate) is not int or replicate not in rng_spec["replicate_indices"]):
        raise ValueError("invalid panel, replicate or index")
    config = protocol["execution_config"]
    regimes = {s["id"]: s for s in config["noise"]["regimes"]}
    if noise_id not in regimes:
        raise ValueError("unregistered noise regime")
    seed = [20260922, rng_spec["stream_version"], replicate, rng_spec["panel_codes"][panel], index]
    shape = tuple(rng_spec["draw_shape"])
    z = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))).standard_normal(shape)
    covariance = noise_covariance(config["observations"]["full_times_hours"], regimes[noise_id])
    if covariance.shape != (z.size, z.size):
        raise ValueError("noise and RNG shapes disagree")
    residual = (np.linalg.cholesky(covariance) @ z.ravel(order="C")).reshape(shape)
    return ResidualPanel(_frozen(residual), JSONRecord({
        "seed": seed, "generator": "PCG64", "numpy_version": np.__version__,
        "panel": panel, "noise_id": noise_id, "units": "ug/m^3",
        "protocol_sha256": binding.full_sha256,
        "standard_normal_sha256": array_hash(z), "residual_sha256": array_hash(residual),
        "generating_covariance_sha256": array_hash(covariance)}))


def _choose_length(scores, lengths):
    """Вернуть индекс наибольшей длины корреляции среди близких минимумов.
    
    Неконечный или неразличимый критерий, а также выбор на границе
    сетки вызывают ``CalibrationFailure``.
    """
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise CalibrationFailure("nonfinite correlation scores")
    best = float(scores.min())
    if np.ptp(scores) <= 1e-8*max(1., abs(best)):
        raise CalibrationFailure("unresolved correlation score contrast")
    eligible = np.flatnonzero(scores-best <= 1e-12+1e-10*abs(best))
    index = int(eligible[-1])
    if index in (0, len(lengths)-1):
        raise CalibrationFailure("correlation length at grid endpoint")
    return index


def covariance_guards(covariance, guards):
    """Проверить численную пригодность матрицы ковариации.

    Parameters
    ----------
    covariance : array_like, shape (n, n)
        Непустая матрица ковариации; вход не корректируется.
    guards : dict of str to float
        Допуски симметрии, ошибки Холецкого, отбеливания и максимального
        спектрального числа обусловленности.

    Returns
    -------
    diagnostics : dict of str to float
        Относительные ошибки симметрии и Холецкого, спектральное отклонение
        отбелённой матрицы от единичной и число обусловленности.

    Raises
    ------
    CalibrationFailure
        Матрица не положительно определена либо диагностика превышает
        заданные допуски.
    """
    matrix = finite_real_array(covariance, "covariance")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise CalibrationFailure("nonempty square covariance required")
    scale = np.linalg.norm(matrix, 2)
    if scale <= 0:
        raise CalibrationFailure("zero covariance")
    symmetry = np.linalg.norm(matrix-matrix.T, 2)/scale
    if symmetry > guards["relative_symmetry"]:
        raise CalibrationFailure("covariance symmetry guard")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise CalibrationFailure("covariance is not SPD") from exc
    condition = np.linalg.cond(matrix, 2)
    backward = np.linalg.norm(factor @ factor.T-matrix, 2)/scale
    left = solve_triangular(factor, matrix, lower=True)
    whitened = solve_triangular(factor, left.T, lower=True).T
    whitening = np.linalg.norm(whitened-np.eye(len(matrix)), 2)
    result = dict(relative_symmetry=float(symmetry), cond2=float(condition),
                  cholesky_backward=float(backward), whitening_spectral=float(whitening))
    limits = dict(guards, cond2=guards["cond2_max"])
    if any(not np.isfinite(v) or v > limits[k] for k, v in result.items()):
        raise CalibrationFailure("covariance numerical guard")
    return result


@dataclass(frozen=True)
class CalibratedCovariance:
    """Ковариация полной панели наблюдений и описание её оценки.

    Parameters
    ----------
    covariance : ndarray, shape (n_observations, n_observations)
        Матрица в (мкг/м³)²; порядок строк: пост, затем время.
    provenance : JSONRecord
        Семейство ковариации, оценки параметров и диагностика.
    """

    covariance: np.ndarray
    provenance: JSONRecord

    def restrict(self, rows, *, layout, guards):
        """Построить метрику ошибок для выбранного набора наблюдений.
        
        Parameters
        ----------
        rows : array_like of int
            Различные индексы наблюдений в исходной ковариации.
        layout : ArraySpace
            Пространство выбранных наблюдений в порядке ``rows``.
        guards : dict
            Допуски численной проверки выбранной ковариации.
        
        Returns
        -------
        metric : CovarianceMetric
            Метрика обратной ковариации и отбеливания выбранных наблюдений.
        diagnostics : dict
            Численные характеристики этой подматрицы.
        """
        indices = np.asarray(rows)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu"
                or len(indices) != layout.size or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(self.covariance))):
            raise ValueError("unique valid selected row indices required")
        selected = self.covariance[np.ix_(indices, indices)]
        diagnostics = covariance_guards(selected, guards)
        return covariance_metric(selected, layout=layout), diagnostics


def fit_covariance(residuals, times, *, family, specification):
    """Оценить ковариацию ошибок с известным нулевым средним.
    
    Parameters
    ----------
    residuals : array_like, shape (n_panels, n_stations, n_times)
        Независимые калибровочные панели ошибок концентрации в мкг/м³.
    times : array_like, shape (n_times,)
        Возрастающие моменты наблюдений в часах.
    family : {'W01', 'W02', 'W03'}
        Общая дисперсия, отдельные дисперсии постов или отдельные дисперсии
        с экспоненциальной временной корреляцией соответственно.
    specification : dict
        Число панелей, границы стандартных отклонений, сетка длин корреляции
        и численные допуски.
    
    Returns
    -------
    calibrated : CalibratedCovariance
        Блочно-диагональная ковариация в (мкг/м³)² и описание её оценки.
    
    Raises
    ------
    CalibrationFailure
        Не выполнены условия калибровки или численные допуски.
    """
    e = finite_real_array(residuals, "calibration residuals")
    times = finite_real_array(times, "calibration times")
    if (e.ndim != 3 or e.shape[0] != specification["panels"]["calib_noise"]
            or e.shape[1] == 0 or e.shape[2] != len(times)
            or times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)):
        raise ValueError("full independent calibration panels and increasing times required")
    if family not in ("W01", "W02", "W03"):
        raise ValueError("unregistered calibration family")
    variances = np.mean(e**2, axis=(0, 2))
    if family == "W01":
        variances[:] = np.mean(e**2)
    lower, upper = specification["sd_bounds"]
    if (not np.isfinite(variances).all() or np.any(np.sqrt(variances) < lower)
            or np.any(np.sqrt(variances) > upper)):
        raise CalibrationFailure("calibrated SD outside bounds; no clipping")
    blocks, fitted_lengths, score_paths = [], [], []
    lags = np.abs(times[:, None]-times[None, :])
    lengths = finite_real_array(specification["ell_grid_hours"], "ell grid")
    if (lengths.ndim != 1 or len(lengths) < 3 or np.any(lengths <= 0)
            or np.any(np.diff(lengths) <= 0)):
        raise ValueError("ordered positive correlation grid required")
    if family == "W03" and len(np.unique(np.round(lags[lags > 0], 12))) < 3:
        raise CalibrationFailure("fewer than three distinct nonzero time lags")
    # Каждая точка сетки корреляции раскладывается один раз для общих оценок постов.
    correlations = []
    if family == "W03":
        for length in lengths:
            correlation = np.exp(-lags/length)
            factor = np.linalg.cholesky(correlation)
            correlations.append((correlation, factor, 2*np.log(np.diag(factor)).sum()))
    for station, variance in enumerate(variances):
        if family != "W03":
            blocks.append(variance*np.eye(len(times)))
            continue
        scores = np.array([.5*(len(e)*logdet+np.sum(
            solve_triangular(factor, e[:, station, :].T, lower=True)**2)/variance)
            for _, factor, logdet in correlations])
        chosen = _choose_length(scores, lengths)
        blocks.append(variance*correlations[chosen][0])
        fitted_lengths.append(float(lengths[chosen]))
        score_paths.append(scores.tolist())
    covariance = block_diag(*blocks)
    diagnostics = covariance_guards(covariance, specification["guards"])
    return CalibratedCovariance(_frozen(covariance), JSONRecord({
        "family": family, "status": "accepted", "units": "(ug/m^3)^2",
        "method": "zero-mean moments; W03 adds conditional correlation grid fit",
        "residuals_sha256": array_hash(e), "times_sha256": array_hash(times),
        "settings_sha256": JSONRecord(specification).sha256,
        "covariance_sha256": array_hash(covariance), "station_variances": variances.tolist(),
        "ell_hours": fitted_lengths, "correlation_scores": score_paths, "guards": diagnostics}))

