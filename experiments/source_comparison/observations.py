"""Пространственные и временные веса наблюдений ADR-модели."""
from dataclasses import dataclass

import numpy as np

from adrkit.spaces import ArraySpace


@dataclass(frozen=True)
class ObservationDesign:
    """Пространственные и временные веса календаря наблюдений.

    Parameters
    ----------
    space_weights : ndarray, shape (n_stations, n_nodes)
        Безразмерные веса пространственных узлов с площадью ячейки.
    time_weights : ndarray, shape (n_times, n_state_times)
        Безразмерные веса временных узлов состояния.
    space : ArraySpace
        Одномерное пространство концентраций в мкг/м³; порядок: пост, затем
        время.
    physical_times : tuple of float
        Моменты наблюдений в часах физического времени.
    """

    space_weights: np.ndarray
    time_weights: np.ndarray
    space: ArraySpace
    physical_times: tuple


def build_observations(binding, solver, *, regime):
    """Построить пространственные веса и временной выбор наблюдений.

    Parameters
    ----------
    binding : ProtocolBinding
        Протокол с расположением постов и календарём наблюдений.
    solver : BoundedSolver
        Решатель, задающий пространственные узлы и временную сетку состояния.
    regime : str
        Идентификатор календаря наблюдений в протоколе.

    Returns
    -------
    design : ObservationDesign
        Гауссовы пространственные веса, выбор временных узлов и пространство
        результата. Наблюдения упорядочены по постам, затем по времени.

    Notes
    -----
    Пространственные веса включают площадь ячейки. Гауссово ядро
    на конечной области не перенормируется.
    """
    config = binding.document.to_dict()["execution_config"]
    spec = config["observations"]
    regimes = {r["id"]: r for r in spec["regimes"]}
    if regime not in regimes:
        raise ValueError(f"unregistered observation regime: {regime}")
    ticks = np.asarray(regimes[regime]["time_indices_zero_based"], dtype=int)
    times = np.asarray(spec["full_times_hours"])[ticks]
    shifted = times-config["model"]["extended_start_hours"]
    indices = np.rint(shifted/solver.dt).astype(int)
    if (np.any(indices < 0) or np.any(indices > solver.nt)
            or not np.allclose(solver.times[indices], shifted, rtol=0, atol=1e-12)):
        raise ValueError("observations must align with the shifted solver clock")
    rho = spec["spatial_standard_deviation_km"]
    spatial = np.array([solver.cell_area*np.exp(
        -np.sum((solver.xy-center)**2, axis=1)/(2*rho**2))/(2*np.pi*rho**2)
        for center in np.asarray(spec["centers_km"])])
    temporal = np.zeros((len(indices), solver.nt+1))
    temporal[np.arange(len(indices)), indices] = 1.
    row_ids = tuple(f"{station}/tick_{j+1}" for station in spec["station_names"] for j in ticks)
    if len(row_ids) != regimes[regime]["rows"]:
        raise ValueError("protocol row count mismatch")
    space = ArraySpace((len(row_ids),), axes=("observation",),
                       coordinates=(row_ids,), units="ug/m^3")
    spatial.flags.writeable = temporal.flags.writeable = False
    return ObservationDesign(spatial, temporal, space, tuple(times))
