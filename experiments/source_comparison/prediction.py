"""Прогноз концентрации для полулинейного сравнения профилей источника."""
import numpy as np

from adrkit.backends.cached import CachedSolver
from experiments.source_comparison.observations import build_observations
from adrkit.observations.grid import GridObservation
from adrkit.predictions import GridPrediction, solver_state_space


def state_solver(protocol, settings, spacing, steps):
    """Создать решатель с заданным пространственным и временным шагом.

    Parameters
    ----------
    protocol : ProtocolBinding
        Протокол с коэффициентами ADR и временным горизонтом.
    settings : dict
        Параметры прямоугольной области; границы заданы в км.
    spacing : float
        Положительный пространственный шаг в км, согласованный с областью.
    steps : int
        Число временных шагов на горизонте протокола.

    Returns
    -------
    solver : CachedSolver
        Решатель концентрации на выбранной сетке.
    """

    spec = protocol.document.to_dict()["execution_config"]["model"]
    model = {k: spec[k] for k in ("diffusion", "velocity", "linear_loss",
        "reaction_gamma", "reaction_c_star", "source_position", "background")}
    model["horizon"] = spec["backend_horizon_hours"]
    bounds = np.asarray(settings["bounds_km"])
    counts = np.diff(bounds.reshape(2, 2), axis=1).ravel()/spacing
    if not np.allclose(counts, np.rint(counts), atol=1e-12, rtol=0):
        raise ValueError("Domain must align with the spatial grid")
    return CachedSolver(model, interior_points=np.rint(counts).astype(int)-1,
                        time_steps=steps, domain=bounds)


class Prediction(GridPrediction):
    """Прогноз концентрации для кусочно-линейного источника.

    Parameters
    ----------
    protocol : ProtocolBinding
        Протокол модели, масштаба интенсивности и наблюдений.
    solver : CachedSolver
        Решатель состояния на выбранной сетке.
    basis : P1Basis
        Базис, использованный при создании load; узлы заданы в часах,
        коэффициенты безразмерны.
    load : IntervalAverageLoad
        Нагрузка на временных шагах решателя с известной предысторией.

    Attributes
    ----------
    basis : P1Basis
        Базис безразмерных узловых значений источника; узлы заданы в часах.
    observations : ObservationDesign
        Веса и пространство концентраций в мкг/м³.

    Notes
    -----
    Решатель и нагрузка копируются базовым GridPrediction. Свойство trajectory
    возвращает только неизменяемый StateCertificate.
    """
    def __init__(self, protocol, solver, basis, load):
        self.basis = basis
        self.observations = build_observations(protocol, solver, regime="dense")
        o = self.observations
        observation = GridObservation(o.space_weights, o.time_weights,
            domain=solver_state_space(solver, state_unit="ug/m^3"), codomain=o.space)
        super().__init__(solver, load, observation)
