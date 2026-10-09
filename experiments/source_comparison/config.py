"""Чтение научных настроек с проверкой контрольной суммы."""
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from adrkit.config.validation import JSONRecord, sha256 as validate_sha256, strict_json
from adrkit.errors import ConfigError

@dataclass(frozen=True)
class ProtocolBinding:
    """Прочитанный протокол и SHA-256 исходных байтов файла.

    Parameters
    ----------
    document : JSONRecord
        Документ протокола сравнения источников.
    full_sha256 : str
        Шестнадцатеричный хеш файла, проверенный при загрузке.
    """

    document: JSONRecord
    full_sha256: str

def load_protocol(path, *, expected_sha256):
    """Прочитать протокол сравнения с заданным хешем файла.

    Parameters
    ----------
    path : str or path-like
        Путь к JSON-файлу протокола.
    expected_sha256 : str
        Ожидаемый SHA-256 исходных байтов файла.

    Returns
    -------
    binding : ProtocolBinding
        Протокол с проверенными хешем, схемой и обязательными разделами.

    Raises
    ------
    ConfigError
        Хеш, структура или схема протокола не совпадают с ожидаемыми.
    """
    validate_sha256(expected_sha256)
    raw = Path(path).read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ConfigError("Source protocol SHA256 mismatch")
    document = JSONRecord(strict_json(raw))
    protocol = document.to_dict()
    sections = {"execution_config", "source_strata", "reproducibility",
                "calibration_and_splits", "solver", "selector"}
    if (protocol.get("schema") != "ym2026.source_protocol.v1"
            or set(protocol) != sections | {"schema"}
            or any(type(protocol[name]) is not dict for name in sections)):
        raise ConfigError("source protocol schema or required sections differ")
    binding = ProtocolBinding(document, digest)
    execution_parameters(binding)
    return binding


def execution_parameters(protocol):
    """Вернуть проверенные масштаб источника, масштаб H¹ и число панелей.

    Parameters
    ----------
    protocol : ProtocolBinding
        Загруженный протокол; дублируемые объявления tau_hours и calib_noise
        должны совпадать.

    Returns
    -------
    dict
        q_reference в C·км²/ч, tau_hours в часах и положительное целое
        calibration_panels. Возвращается новая запись.

    Raises
    ------
    ConfigError
        Значения отсутствуют, недопустимы либо расходятся между разделами.
    """
    document = protocol.document.to_dict()
    try:
        spec = document["execution_config"]
        q_reference = spec["basis"]["Qref"]
        tau = spec["basis"]["tau_hours"]
        selector_tau = document["selector"]["tau_hours"]
        count = spec["calibration"]["panels"]["calib_noise"]
        split_count = document["calibration_and_splits"]["panels"]["calib_noise"]
    except (KeyError, TypeError) as error:
        raise ConfigError("Qref, tau_hours and calib_noise declarations are required") from error
    for name, value in (("Qref", q_reference), ("basis.tau_hours", tau),
                        ("selector.tau_hours", selector_tau)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{name} must be a finite positive real number")
    if tau != selector_tau:
        raise ConfigError("basis and selector tau_hours declarations differ")
    if (type(count) is not int or type(split_count) is not int
            or count <= 0 or split_count <= 0):
        raise ConfigError("calib_noise must be a positive integer in both declarations")
    if count != split_count:
        raise ConfigError("calib_noise panel declarations differ")
    return dict(q_reference=float(q_reference), tau_hours=float(tau),
                calibration_panels=count)
