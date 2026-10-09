"""Общие искусственные входы коротких проверок КрАЗ."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def short_inputs(tmp_path, project_root):
    """Создать малую ADR-задачу и независимые искусственные CSV."""
    simulation = tmp_path / "simulation.json"
    protocol = project_root / "experiments/source_comparison/configs/protocol.json"
    shutil.copyfile(protocol, tmp_path / "protocol.json")
    settings = json.loads((protocol.parent / "experiment.json").read_bytes())
    settings.update(output="unused-simulation-output", sources=["SF01-PG10"],
                    truth_spacing_km=2.0, truth_steps=84, protected_roots=[])
    simulation.write_bytes((json.dumps(settings, ensure_ascii=False, indent=3)
                            + "\n\n").encode("utf-8"))
    data = tmp_path / "raw"
    for folder in ("sev", "pes", "slc", "krz"):
        (data / folder).mkdir(parents=True)
        for year in range(2019, 2023):
            times = pd.date_range(f"{year}-01-01", periods=19 if year in (2019, 2022) else 0,
                                  freq="20min")
            values = 20.0 + np.arange(len(times), dtype=float)
            frame = pd.DataFrame({"date": times.strftime("%Y-%m-%d"),
                                  "time": times.strftime("%H:%M"),
                                  "t": values * 0.0, "p": values * 0.0 + 1000.0,
                                  "h": values * 0.0 + 50.0, "ws": values * 0.0 + 1.0,
                                  "wd": values * 0.0, "pm25": values})
            frame.to_csv(data / folder / f"{year}.csv", sep=";", index=False)
    config = tmp_path / "kraz.json"
    cfg = dict(scope="Искусственная проверка расчёта КрАЗ",
               simulation_config="simulation.json", data_root="raw", output="result",
               amplitude_factors=[0.5, 1.0], fit_points=6, minimum_fit_range=1.0,
               window_hours=3, window_selection="minimum fitting score",
               splits={"exploration": ["2019-01-01", "2019-01-02"],
                       "later_period": ["2022-01-01", "2022-01-02"]})
    config.write_bytes((json.dumps(cfg, ensure_ascii=False, indent=3)
                       + "\n\n").encode("utf-8"))
    return config, simulation, tmp_path / "result"

