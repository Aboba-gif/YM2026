"""Временные веса мгновенных и усреднённых наблюдений E06."""
from __future__ import annotations

import numpy as np

from adrkit.loads import interval_hat_integrals
from adrkit.sources import P1Basis
from adrkit.predictions import (
    GridPrediction as ProjectedPrediction, StateCertificate,
    grid_observation, solver_state_space,
)


def temporal_weights(state_times, observation_times, *, origin_hours,
                     kind="snapshot", window_hours=None):
    """Построить временные веса наблюдения кусочно-линейного состояния.
    
    Parameters
    ----------
    state_times : array_like, shape (n_states,)
        Возрастающие относительные времена в часах, начиная с нуля.
    observation_times : array_like, shape (n_observations,)
        Возрастающие физические времена наблюдений в часах.
    origin_hours : float
        Физическое начало шкалы ``state_times`` в часах.
    kind : {'snapshot', 'average'}, optional
        Выбор состояния в узле или среднее по предшествующему окну.
    window_hours : float or None, optional
        Положительная длительность окна для 'average'; для 'snapshot' — None.
    
    Returns
    -------
    ndarray, shape (n_observations, n_states)
        Безразмерные веса в порядке входных времён. Для 'snapshot'
        момент совпадает с узлом, для 'average' всё окно лежит в шкале
        состояния, а веса интегрируют P1-интерполянт и делятся на длину окна.
    """
    if any(np.iscomplexobj(v) for v in (state_times, observation_times, origin_hours)):
        raise ValueError("Time arguments must be real")
    states = np.asarray(state_times, dtype=float)
    times = np.asarray(observation_times, dtype=float)
    origin = np.asarray(origin_hours, dtype=float)
    if (states.ndim != 1 or len(states) < 2 or not np.isfinite(states).all()
            or states[0] != 0 or np.any(np.diff(states) <= 0)
            or times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or origin.shape != ()
            or not np.isfinite(origin)):
        raise ValueError("Increasing finite clocks and scalar origin required")
    physical = states + float(origin)
    if kind == "snapshot":
        if window_hours is not None:
            raise ValueError("A snapshot has no averaging window")
        result = np.zeros((len(times), len(states)))
        for j, time in enumerate(times):
            index = int(np.argmin(np.abs(physical-time)))
            if not np.isclose(physical[index], time, rtol=0, atol=1e-12):
                raise ValueError("Snapshot does not coincide with the state clock")
            result[j, index] = 1.
        return result
    if kind != "average":
        raise ValueError("Temporal kind must be snapshot or average")
    if isinstance(window_hours, (bool, np.bool_)) or np.iscomplexobj(window_hours):
        raise ValueError("Averaging window must be real and positive")
    width = np.asarray(window_hours, dtype=float)
    if width.shape != () or not np.isfinite(width) or width <= 0:
        raise ValueError("Averaging window must be a positive finite scalar")
    width = float(width)
    if np.any(times-width < physical[0]-1e-12) or np.any(times > physical[-1]+1e-12):
        raise ValueError("Entire averaging interval must be inside the state clock")
    basis = P1Basis(physical, time_unit="h")
    result = np.vstack([(1./width)*interval_hat_integrals(basis, [t-width, t])[0]
                        for t in times])
    if np.any(result < 0) or not np.allclose(result.sum(axis=1), 1., rtol=0, atol=2e-12):
        raise ValueError("Temporal integration failed positivity or coverage")
    return result


