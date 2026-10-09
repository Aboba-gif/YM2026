"""Параметры закреплённого численного опыта восстановления источника."""
from __future__ import annotations

import numpy as np


SCHEMA = "ym2026.research_validation.v1"


DENSE_TIMES = np.arange(1, 73, dtype=float)/24


PRIMARY_TICKS = np.arange(7, 72, 8)


PRIMARY_ROWS = np.concatenate([s*72+PRIMARY_TICKS for s in range(4)])


PANEL_CODES = dict(calibration=101, fit=211, selection=307, test=401, mask=503)


BASE_EXPONENTS = tuple(float(x) for x in np.arange(-8., 4.01, .5))


def template_config(source_mass):
    """Создать исходную конфигурацию восстановления источника.

    Parameters
    ----------
    source_mass : float
        Положительный интеграл каждого профиля на 0–3 часах в C·км²; C —
        единица концентрации.

    Returns
    -------
    spec : dict
        Конфигурация с пустым списком условий и frozen=False; для запуска
        нужны явные условия и фиксация.
    """
    mass = float(source_mass)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Positive source mass required")
    return dict(schema=SCHEMA, frozen=False, study="research_validation",
        output="../../../../YM2026-results/source_recovery",
        source_protocol=dict(path="../../source_comparison/configs/protocol.json",
            sha256="b0ea2697407f905ea78334d4d1d10d74c0b0ed205845451974bc4d09830e5078"),
        input_files=[], source_mass=mass, Qref=100.,
        sources=["PG10", "SB150", "EC04", "EC06", "NEW-J2", "NEW-S2"],
        replicates=list(range(1, 9)),
        stream=dict(seed=20260926, version=2),
        model=dict(diffusion=.018, velocity=[-.72, -.144], linear_loss=1/72,
            reaction_gamma=1/72, reaction_c_star=1., source_position=[0., 0.],
            background=0., horizon=3.5, origin_hours=-.5),
        observations=dict(station_names=["Severny", "Peschanka", "Soloncy", "KrAZ"],
            centers_km=[[-4.807498961147989, -2.0443267416397575],
                        [4.071301019331992, -.4486151101289735],
                        [-10.93028096310831, -2.14041318690405], [0., 0.]],
            spatial_standard_deviation_km=1.),
        grids=dict(G0=dict(bounds_km=[-14., 8., -6., 6.], spacing_km=.25, steps=336),
            Gt=dict(bounds_km=[-14., 8., -6., 6.], spacing_km=.25, steps=672),
            Gx=dict(bounds_km=[-14., 8., -6., 6.], spacing_km=.125, steps=336),
            Gxt=dict(bounds_km=[-14., 8., -6., 6.], spacing_km=.125, steps=672)),
        truth_grid="G0",
        noise=dict(corr=dict(type="station_exponential", station_sd=[1.,1.5,2.,1.],
                           ell_hours=[.25,.25,.25,.25]),
            iid_high=dict(type="station_iid", station_sd=[1.,1.5,2.,1.]),
            iid_low=dict(type="station_iid", station_sd=[.1,.15,.2,.1])),
        calibration=dict(panels=dict(calib_noise=32), sd_bounds=[.001, 100.],
            ell_grid_hours=np.geomspace(1/60, 1.5, 33).tolist(),
            guards=dict(relative_symmetry=1e-12, cond2_max=1e10,
                        cholesky_backward=1e-12, whitening_spectral=1e-8)),
        solver=dict(kkt_tolerance=1e-6, max_iterations=80, forward_tolerance=1e-11),
        alpha=dict(policy="fixed_common_grid", exponents=list(BASE_EXPONENTS),
                   tie_atol=1e-12, tie_rtol=1e-10),
        conditions=[], single_fits=[])


def validate_config(spec, *, require_frozen=True):
    """Проверить научные условия конфигурации восстановления.

    Parameters
    ----------
    spec : dict
        Конфигурация источников, сеток, шума, штрафов и конечного набора
        условий.
    require_frozen : bool, optional
        Требовать frozen=True; по умолчанию True.

    Raises
    ------
    ValueError
        Не соблюдены зарегистрированные условия, ссылки или правила выбора
        регуляризации.
    """

    from .checkpoints import canonical_hash
    from .panels import selected_rows
    from .driver import _conditions
    if spec.get("schema") != SCHEMA or (require_frozen and spec.get("frozen") is not True):
        raise ValueError("New explicitly frozen research configuration required")
    if spec["model"]["background"] != 0 or spec["model"]["origin_hours"] != -.5 or spec["model"]["horizon"] != 3.5:
        raise ValueError("Registered zero-background history/unknown clock is [-.5,3]")
    if any(type(spec["stream"][key]) is not int or spec["stream"][key] < 0 for key in ("seed","version")):
        raise ValueError("Nonnegative integer seed/version required")
    if not set(spec["sources"]) <= {"PG10","SB150","EC04","EC06","NEW-J2","NEW-S2"}:
        raise ValueError("Only registered source IDs are available")
    if spec["calibration"]["panels"]["calib_noise"] != 32:
        raise ValueError("Exactly32 independent primary calibration panels required")
    if (tuple(spec["alpha"]["exponents"]) != BASE_EXPONENTS or
        spec["alpha"].get("policy") != "fixed_common_grid" or
        {"extension_rounds","cap"} & spec["alpha"].keys()):
        raise ValueError("Reviewed fixed25-node grid [-8,4] with step .5 required")
    if spec["alpha"]["tie_atol"] < 0 or spec["alpha"]["tie_rtol"] < 0:
        raise ValueError("Nonnegative selection tie tolerances required")
    if not spec["conditions"]:
        raise ValueError("Explicit finite conditions required")
    if spec["truth_grid"] not in spec["grids"]:
        raise ValueError("Unknown default truth grid")
    required = {"id","sources","replicates","weight","penalties","grid","nodes","noise",
                "tau_hours","truth_gamma","inverse_gamma","availability","mask","temporal_H","relocation_km"}
    ids = []
    for condition in spec["conditions"]:
        if required-condition.keys():
            raise ValueError(f"Missing condition fields: {sorted(required-condition.keys())}")
        ids.append(condition["id"])
        if condition["weight"] not in ("W01","W02","W03","Woracle"):
            raise ValueError("Unknown weighting family")
        if not set(condition["penalties"]) <= {"L2","H1"} or not condition["penalties"]:
            raise ValueError("L2/full H1 penalties required")
        if not set(condition["sources"]) <= set(spec["sources"]) or not set(condition["replicates"]) <= set(spec["replicates"]):
            raise ValueError("Condition source/replicate not registered")
        if condition["grid"] not in spec["grids"] or condition["noise"] not in spec["noise"]:
            raise ValueError("Condition references unknown grid/noise")
        if condition.get("truth_grid",spec["truth_grid"]) not in spec["grids"]:
            raise ValueError("Condition references unknown truth grid")
        if condition["nodes"] not in (73,145,289) or condition["tau_hours"] <= 0:
            raise ValueError("Registered basis and positive tau required")
        if condition.get("fit_regime","primary") not in ("primary","dense"):
            raise ValueError("Unknown fit calendar")
        if condition["temporal_H"] not in ("snapshot","average20"):
            raise ValueError("Unknown temporal H")
        if min(condition["truth_gamma"],condition["inverse_gamma"]) < 0:
            raise ValueError("Nonnegative reaction required")
        selected_rows(condition,spec["stream"],1)
    if len(ids) != len(set(ids)) or any(not isinstance(i,str) or not i or "/" in i or "\\" in i for i in ids):
        raise ValueError("Unique simple condition IDs required")
    singles = spec.get("single_fits",[])
    if len({s["id"] for s in singles}) != len(singles):
        raise ValueError("Unique single-fit IDs required")
    for single in singles:
        if single["condition"] not in ids or single["baseline_condition"] not in ids or single["kind"] not in ("fixed_alpha","zero_start"):
            raise ValueError("Single fit must reference configured conditions and kind")
        if single["penalty"] not in ("L2","H1") or not set(single["sources"]) <= set(spec["sources"]) or not set(single["replicates"]) <= set(spec["replicates"]):
            raise ValueError("Single fit has unregistered arm/source/replicate")
        for name in single["sources"]:
            for replicate in single["replicates"]:
                active = {c["id"]:c for c in _conditions(spec,name,replicate)}
                if single["baseline_condition"] not in active or single["penalty"] not in active[single["baseline_condition"]]["penalties"]:
                    raise ValueError("Single-fit baseline path absent in this source/replicate")
    if any(type(r) is not int or r < 1 for r in spec["replicates"]):
        raise ValueError("Main replicate IDs start at1; technical pilot excluded")
    canonical_hash(spec)  # Каноническая запись отклоняет неконечные числа.
