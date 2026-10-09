"""Запись и проверка состояния одного расчёта восстановления источника."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from adrkit.config.validation import JSONRecord
from .config import BASE_EXPONENTS


def canonical_hash(value):
    """Вернуть SHA-256 канонической записи JSONRecord."""

    return JSONRecord(value).sha256


def file_hash(path):
    """Вернуть шестнадцатеричный SHA-256 байтов указанного файла."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    """Записать JSON через уникальный временный файл с fsync и заменой цели."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Checkpoint:
    """Файл состояния восстановления для одного источника и реализации шума.

    Parameters
    ----------
    path : path-like
        Файл JSON для создания или продолжения расчёта.
    bindings : dict
        Идентификаторы источника, реализации и контрольные суммы входов.
    expected_paths : iterable of str
        Идентификаторы предусмотренных последовательностей оценок.

    Attributes
    ----------
    path : Path
        Путь к файлу состояния.
    record : dict
        Текущие результаты и стадия расчёта.
        Изменения записываются методом `save`.

    Notes
    -----
    Файл изменяет один процесс. При запуске опыта run_experiment удерживает
    блокировку группы от чтения состояния до завершения записи.
    """
    def __init__(self, path, bindings, expected_paths):
        self.path = Path(path)
        if self.path.exists():
            self.record = json.loads(self.path.read_text(encoding="utf-8"))
            if self.record["bindings"] != bindings or self.record["expected_paths"] != list(expected_paths):
                raise ValueError("Checkpoint bindings changed; use a new output directory")
        else:
            self.record = dict(bindings=bindings, expected_paths=list(expected_paths),
                paths={}, stage="estimating", exponents=list(BASE_EXPONENTS))
            self.save()

    def save(self):
        """Записать текущее состояние в файл с заменой через временный файл."""

        atomic_json(self.path, self.record)

    def seal(self):
        """Зафиксировать хеш завершённых записей выбора.

        Returns
        -------
        digest : str
            SHA-256 записей paths; также сохраняется в selection_seal.

        Raises
        ------
        ValueError
            Набор путей неполон, запись не завершена или ранее зафиксированные
            данные изменились.
        """

        if set(self.record["paths"]) != set(self.record["expected_paths"]):
            raise ValueError("Cannot expose test/truth before all paths are recorded")
        if not all(r.get("finalized") for r in self.record["paths"].values()):
            raise ValueError("Fixed-grid or single-fit records are unfinished")
        payload = {k:v for k,v in self.record["paths"].items()}
        digest = canonical_hash(payload)
        if "selection_seal" in self.record and self.record["selection_seal"] != digest:
            raise ValueError("Selected records changed after sealing")
        self.record.update(selection_seal=digest, stage="sealed")
        self.save()
        return digest

    def require_sealed(self):
        """Проверить стадию и хеш зафиксированного выбора.

        Raises
        ------
        ValueError
            Стадия не sealed/scored либо записи paths изменены после фиксации.
        """

        if self.record.get("stage") not in ("sealed", "scored"):
            raise ValueError("Scoring is sealed until every path is finalized")
        if canonical_hash(self.record["paths"]) != self.record["selection_seal"]:
            raise ValueError("Selection seal mismatch")
