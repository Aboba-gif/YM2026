"""Чтение исходных CSV четырёх постов и сбор панели PM₂.₅."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

SITES = ("Severny", "Peschanka", "Soloncy", "KrAZ")
FOLDERS = dict(zip(SITES, ("sev", "pes", "slc", "krz")))
CHANNELS = ("t", "p", "h", "ws", "wd", "pm25")


def file_sha(path: Path) -> str:
    """Вычислить SHA-256 байтов файла.

    Parameters
    ----------
    path : Path
        Путь к читаемому файлу.

    Returns
    -------
    str
        Шестнадцатеричный SHA-256 полного содержимого.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_network(data_root: Path):
    """Прочитать исходные CSV четырёх постов за 2019–2022 годы.

    Parameters
    ----------
    data_root : Path
        Каталог с четырьмя годовыми CSV в каждой подпапке `FOLDERS`.

    Returns
    -------
    index : DatetimeIndex
        Общая сетка с шагом 20 минут от начала 2019 года до конца 2022 года.
    frames : dict of str to DataFrame
        Исходные каналы каждого поста, выровненные по `index`. NaN обозначает
        пропуск, `row_present` отличает отсутствующую строку от строки с
        пропусками значений.
    manifest : list of dict
        Пути относительно data_root, размеры, числа исходных строк и SHA-256
        прочитанных файлов.
    """
    index = pd.date_range("2019-01-01", "2023-01-01", freq="20min", inclusive="left")
    frames, manifest = {}, []
    for site, folder in FOLDERS.items():
        parts = []
        paths = sorted((data_root / folder).glob("*.csv"))
        if len(paths) != 4:
            raise ValueError(f"Expected four original yearly CSVs for {site}")
        for path in paths:
            content = path.read_bytes()
            part = pd.read_csv(BytesIO(content), sep=";")
            date_text, time_text = part.pop("date").astype(str), part.pop("time").astype(str)
            if date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
                timestamp = pd.to_datetime(date_text + " " + time_text, format="ISO8601")
            elif date_text.str.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}").all():
                timestamp = pd.to_datetime(date_text + " " + time_text, format="%d.%m.%Y %H:%M")
            else:
                raise ValueError("Undocumented raw date format")
            part.index = timestamp
            for channel in CHANNELS:
                part[channel] = pd.to_numeric(part[channel], errors="raise")
            if not part.index.isin(index).all():
                raise ValueError("Unexpected off-grid or out-of-period timestamp")
            parts.append(part[list(CHANNELS)])
            manifest.append(dict(site=site, file=f"{folder}/{path.name}",
                                 bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), rows=len(part)))
        original = pd.concat(parts).sort_index()
        if not original.index.is_unique:
            raise ValueError("Duplicate raw timestamps require an explicit resolution")
        frame = original.reindex(index)
        frame["row_present"] = index.isin(original.index)
        values = frame[list(CHANNELS)].to_numpy()
        if np.isinf(values).any() or (frame.pm25.dropna() < 0).any():
            raise ValueError("Nonfinite/negative raw observation requires explicit QC")
        frames[site] = frame
    return index, frames, manifest


def pm25_matrix(frames):
    """Собрать исходные ряды PM₂.₅ в порядке SITES.

    Parameters
    ----------
    frames : dict of str to DataFrame
        Выровненные по общему индексу таблицы с колонкой `pm25` для всех
        постов `SITES`.

    Returns
    -------
    ndarray, shape (n, 4)
        Значения в единицах исходного числового экспорта; NaN и порядок строк сохраняются.
    """

    return np.column_stack([frames[site].pm25.to_numpy() for site in SITES])
