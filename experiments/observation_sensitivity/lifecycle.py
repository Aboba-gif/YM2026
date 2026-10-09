"""Сохранение и продолжение контрольных записей групп E06.

``Checkpoint`` фиксирует выбор оценок перед вычислением проверочных
ошибок. Режимы восстановления подготовленной записи описаны в README.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import time

from adrkit.config.validation import canonical_bytes, digest, strict_json
from .design import ALPHA_EXPONENTS
from experiments.file_locks import FileLock, FileLockBusy


SCHEMA = "ym2026.observation_sensitivity.checkpoint"
VERSION = 1
_EXPONENT_KEYS = tuple(str(x) for x in ALPHA_EXPONENTS)
_IMMUTABLE = ("schema", "version", "bindings", "expected_paths", "exponents")
_BASE_KEYS = {*_IMMUTABLE, "paths", "stage", "revision", "content_sha256"}


class CheckpointError(ValueError):
    """Неверная структура, изменённая привязка, данные или переход стадии."""


class LeaseBusyError(RuntimeError):
    """Блокировку контрольной записи удерживает другой процесс."""


class PersistenceError(RuntimeError):
    """Ошибка сохранения; численная попытка не повторяется, экземпляр больше не используется."""


class PendingRecoveryRequired(PersistenceError):
    """Подготовленная запись требует проверки и явного завершения сохранения."""


def _same(left, right):
    # Сравнение канонического JSON различает True и 1.
    
    return canonical_bytes(left) == canonical_bytes(right)


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _identity(bindings, expected_paths):
    canonical_bytes(bindings)
    if type(bindings) is not dict:
        raise CheckpointError("bindings must be a JSON object")
    source, replicate = bindings.get("source"), bindings.get("replicate")
    if (type(source) is not str or not re.fullmatch(r"[A-Za-z0-9_-]+", source)
            or type(replicate) is not int or replicate < 1):
        raise CheckpointError("bindings require a source ID and positive integer replicate")
    if (type(expected_paths) is not list or not expected_paths
            or any(type(x) is not str for x in expected_paths)
            or len(set(expected_paths)) != len(expected_paths)):
        raise CheckpointError("expected_paths must be a nonempty ordered list of unique IDs")
    pattern = rf"[A-Za-z0-9_]+/{re.escape(source)}/r{replicate}/(?:L2|H1)"
    if any(re.fullmatch(pattern, x) is None for x in expected_paths):
        raise CheckpointError("path ID must match condition/source/replicate/penalty binding")


def _candidate(candidate, exponent=None):
    if (type(candidate) is not dict or type(candidate.get("accepted")) is not bool
            or not _positive_number(candidate.get("alpha"))):
        raise CheckpointError("candidate needs positive finite alpha and boolean accepted")
    if exponent is not None and (type(candidate.get("exponent")) not in (int, float)
                                 or candidate["exponent"] != exponent):
        raise CheckpointError("candidate exponent does not match the fixed grid key")


def _path(record, *, terminal=False):
    if (type(record) is not dict or type(record.get("finalized")) is not bool
            or type(record.get("candidates")) is not dict):
        raise CheckpointError("path needs boolean finalized and candidates object")
    candidates = record["candidates"]
    if not set(candidates) <= set(_EXPONENT_KEYS):
        raise CheckpointError("candidate key outside the fixed 25-exponent grid")
    reference = record.get("alpha_reference")
    if candidates and not _positive_number(reference):
        raise CheckpointError("candidate path requires positive finite alpha_reference")
    for key, candidate in candidates.items():
        _candidate(candidate, float(key))
        expected_alpha = reference * 10.**float(key)
        if (not _positive_number(expected_alpha)
                or not math.isclose(candidate["alpha"], expected_alpha, rel_tol=2e-14, abs_tol=0.)):
            raise CheckpointError("physical alpha disagrees with alpha_reference and fixed exponent")
    if "calibration_failure" in record:
        if (type(record["calibration_failure"]) is not str or not record["calibration_failure"]
                or candidates or record["finalized"] is not True
                or record.get("procedure_accepted") is not False):
            raise CheckpointError("calibration failure must be terminal, rejected and have no candidates")
    elif record["finalized"] and set(candidates) != set(_EXPONENT_KEYS):
        raise CheckpointError("finalized path requires all 25 attempts, including rejected attempts")
    if terminal and record["finalized"] is not True:
        raise CheckpointError("all expected paths must be finalized before sealing")


def _selection_payload(record):
    return {key: record[key] for key in (*_IMMUTABLE, "paths")}


def _validate(record, bindings, expected_paths, *, check_content=True):
    canonical_bytes(record)
    if type(record) is not dict or not _BASE_KEYS <= set(record):
        raise CheckpointError("incomplete checkpoint envelope")
    optional = {"selection_seal", "scores", "origin"}
    if set(record) - _BASE_KEYS - optional:
        raise CheckpointError("unknown checkpoint envelope fields")
    if record["schema"] != SCHEMA or type(record["version"]) is not int or record["version"] != VERSION:
        raise CheckpointError("unsupported checkpoint schema/version")
    if (not _same(record["bindings"], bindings)
            or not _same(record["expected_paths"], expected_paths)):
        raise CheckpointError("checkpoint bindings or ordered expected paths changed")
    if not _same(record["exponents"], list(ALPHA_EXPONENTS)):
        raise CheckpointError("checkpoint alpha grid changed")
    if type(record["revision"]) is not int or record["revision"] < 0:
        raise CheckpointError("invalid checkpoint revision")
    if "origin" in record:
        origin = record["origin"]
        if (type(origin) is not dict
                or set(origin) != {"artifact_sha256", "content_sha256", "selection_seal"}
                or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in origin.values())):
            raise CheckpointError("origin requires the original artifact, content and selection hashes")
    stage = record["stage"]
    if stage not in ("estimating", "sealed", "scored"):
        raise CheckpointError("invalid checkpoint stage")
    if type(record["paths"]) is not dict or not set(record["paths"]) <= set(expected_paths):
        raise CheckpointError("unplanned path in checkpoint")
    terminal = stage != "estimating"
    for row in record["paths"].values():
        _path(row, terminal=terminal)
    if terminal:
        if set(record["paths"]) != set(expected_paths):
            raise CheckpointError("sealed checkpoint omits expected paths")
        if record.get("selection_seal") != digest(_selection_payload(record)):
            raise CheckpointError("selection seal mismatch")
    elif "selection_seal" in record or "scores" in record:
        raise CheckpointError("no truth/test scores or seal while estimating")
    if "scores" in record:
        if type(record["scores"]) is not dict or not set(record["scores"]) <= set(expected_paths):
            raise CheckpointError("scores must map only expected path IDs")
        if any(type(row) is not dict or not row for row in record["scores"].values()):
            raise CheckpointError("each score must be a nonempty record, including unavailable outcomes")
    if stage == "scored" and set(record.get("scores", {})) != set(expected_paths):
        raise CheckpointError("scored checkpoint must account for every planned path")
    if check_content:
        payload = {key: value for key, value in record.items() if key != "content_sha256"}
        if record["content_sha256"] != digest(payload):
            raise CheckpointError("checkpoint content digest mismatch")


def _append_only(old, new, *, sealing=False):
    if not _same(old.get("origin"), new.get("origin")):
        raise CheckpointError("checkpoint origin cannot change")
    for key in _IMMUTABLE:
        if not _same(old[key], new[key]):
            raise CheckpointError(f"immutable checkpoint field changed: {key}")
    stages = {"estimating": {"estimating", "sealed"} if sealing else {"estimating"},
              "sealed": {"sealed", "scored"}, "scored": {"scored"}}
    if new["stage"] not in stages[old["stage"]]:
        raise CheckpointError("illegal lifecycle transition; use seal() before scoring")
    if not set(old["paths"]) <= set(new["paths"]):
        raise CheckpointError("persisted paths cannot be removed")
    for pid, prior in old["paths"].items():
        current = new["paths"][pid]
        if prior["finalized"] or old["stage"] != "estimating":
            if not _same(prior, current):
                raise CheckpointError("finalized paths cannot change")
        else:
            for key, value in prior.items():
                if key == "finalized":
                    continue
                if key == "candidates":
                    for exponent, candidate in value.items():
                        if exponent not in current[key] or not _same(candidate, current[key][exponent]):
                            raise CheckpointError("persisted candidate cannot change or disappear")
                elif key not in current or not _same(value, current[key]):
                    raise CheckpointError("persisted path provenance cannot change")
    for pid, prior in old.get("scores", {}).items():
        if pid not in new.get("scores", {}) or not _same(prior, new["scores"][pid]):
            raise CheckpointError("persisted scores cannot change or disappear")
    if old["stage"] == "scored" and not _same(old["scores"], new.get("scores")):
        raise CheckpointError("completed scoring cannot change")


class _Lease(FileLock):
    def __init__(self, path, *, shared=False, create=True):
        try:
            super().__init__(path, shared=shared, create=create)
        except FileLockBusy as error:
            raise LeaseBusyError("checkpoint already has an incompatible reader or writer") from error


def _sync_directory(path):
    # Windows fsync сбрасывает временный файл. Переносимого API сброса каталога в Python нет;
    # сохранность при любом отказе питания не гарантируется.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _replace_pending(pending, target, attempts, delay):
    for index in range(attempts):
        try:
            os.replace(pending, target)
            break
        except OSError as error:
            retryable = isinstance(error, PermissionError) or getattr(error, "winerror", None) in (32, 33)
            if not retryable or index + 1 == attempts:
                raise
            time.sleep(delay * 2**index)
    _sync_directory(target.parent)


def _checkpoint_file(value):
    """Проверить отсутствие псевдонимов и единственную ссылку на обычный файл."""
    path = Path(os.path.abspath(value))
    for node in (path, *path.parents):
        if node.is_symlink() or getattr(node, "is_junction", lambda: False)():
            raise CheckpointError("Checkpoint files must not be filesystem aliases")
    if path.resolve() != path:
        raise CheckpointError("Checkpoint files must not be filesystem aliases")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return path
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CheckpointError("Checkpoint storage requires an ordinary file without hard links")
    return path


class Checkpoint:
    """Контрольная запись одной группы с файловой блокировкой.

    Экземпляр поддерживает контекстный менеджер. После ошибки сохранения его
    следует закрыть. Численные попытки при восстановлении сохранения не
    повторяются.

    Parameters
    ----------
    path : str or pathlib.Path
        Путь к файлу группы; родительский каталог создаётся при
        необходимости.
    bindings : dict
        JSON-привязки источника, положительного номера реализации и входных
        данных.
    expected_paths : iterable of str
        Непустой упорядоченный список уникальных идентификаторов той же
        группы.
    recover_pending : bool, optional
        Проверить и завершить подготовленное сохранение при открытии.
    replace_attempts : int, optional
        Положительное число попыток замены файла.
    retry_delay : float, optional
        Начальная неотрицательная задержка повторной замены в секундах.

    Attributes
    ----------
    record : dict
        Изменяемая запись группы. Допустимые добавления сохраняются через
        save; выбор закрепляется через seal.
    """

    def __init__(self, path, bindings, expected_paths, *, recover_pending=False,
                 replace_attempts=5, retry_delay=0.05):
        if (type(replace_attempts) is not int or replace_attempts < 1
                or type(retry_delay) not in (int, float) or not math.isfinite(retry_delay)
                or retry_delay < 0 or type(recover_pending) is not bool):
            raise CheckpointError("positive retry count, finite nonnegative delay and boolean recovery required")
        if isinstance(expected_paths, (str, bytes, set, frozenset, dict)):
            raise CheckpointError("expected_paths must have explicit order")
        expected_paths = list(expected_paths)
        _identity(bindings, expected_paths)
        self._bindings = deepcopy(bindings)
        self._expected = list(expected_paths)
        self.path = _checkpoint_file(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.path.with_name(self.path.name + ".pending")
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._lease = _Lease(self.lock_path)
        self._owner_pid = os.getpid()
        self._poisoned = False
        self._attempts, self._delay = replace_attempts, retry_delay
        self._committed = None
        try:
            _checkpoint_file(self.path)
            _checkpoint_file(self.pending_path)
            if self.path.exists():
                self._committed = self._read(self.path)
            if self.pending_path.exists():
                if not recover_pending:
                    raise PendingRecoveryRequired("pending save exists; inspect and explicitly recover_pending=True")
                pending = self._read(self.pending_path)
                if self._committed is None:
                    if pending["revision"] != 0 or pending["stage"] != "estimating" or pending["paths"]:
                        raise CheckpointError("pending initial checkpoint has invalid history")
                else:
                    if pending["revision"] != self._committed["revision"] + 1:
                        raise CheckpointError("pending revision does not extend committed revision")
                    _append_only(self._committed, pending, sealing=True)
                try:
                    _checkpoint_file(self.path)
                    _checkpoint_file(self.pending_path)
                    _replace_pending(self.pending_path, self.path, self._attempts, self._delay)
                except OSError as error:
                    raise PersistenceError("pending commit recovery failed; no numerical retry") from error
                self._committed = pending
            if self._committed is None:
                self.record = dict(schema=SCHEMA, version=VERSION, bindings=deepcopy(self._bindings),
                    expected_paths=list(self._expected), exponents=list(ALPHA_EXPONENTS),
                    paths={}, stage="estimating", revision=0, content_sha256="")
                self._save(initial=True)
            else:
                self.record = deepcopy(self._committed)
        except BaseException:
            self.close()
            raise

    def _read(self, path):
        record = strict_json(_checkpoint_file(path).read_bytes())
        _validate(record, self._bindings, self._expected)
        return record

    def _active(self):
        if self._owner_pid != os.getpid():
            raise PersistenceError("checkpoint instance cannot be used by an inherited child process")
        if self._lease.fd is None:
            raise PersistenceError("checkpoint is closed")
        if self._poisoned:
            raise PersistenceError("checkpoint save failed; close and inspect/recover, do not reroll attempts")

    def _save(self, *, sealing=False, initial=False):
        self._active()
        candidate = deepcopy(self.record)
        if not initial:
            if (candidate.get("revision") != self._committed["revision"]
                    or not _same(candidate.get("content_sha256"), self._committed["content_sha256"])):
                raise CheckpointError("revision/content digest are managed by Checkpoint")
            _validate(candidate, self._bindings, self._expected, check_content=False)
            _append_only(self._committed, candidate, sealing=sealing)
            candidate["revision"] += 1
        candidate["content_sha256"] = digest({k: v for k, v in candidate.items() if k != "content_sha256"})
        _validate(candidate, self._bindings, self._expected)
        try:
            # Предыдущая незавершённая запись не перезаписывается.
            _checkpoint_file(self.path)
            with _checkpoint_file(self.pending_path).open("xb") as stream:
                stream.write(canonical_bytes(candidate) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            _checkpoint_file(self.path)
            _checkpoint_file(self.pending_path)
            _replace_pending(self.pending_path, self.path, self._attempts, self._delay)
        except (OSError, CheckpointError) as error:
            self._poisoned = True
            raise PersistenceError("checkpoint persistence failed; pending payload retained if created; no numerical retry") from error
        # Сохраняются ссылки driver на record и paths.
        self.record["revision"] = candidate["revision"]
        self.record["content_sha256"] = candidate["content_sha256"]
        self._committed = candidate

    def save(self):
        """Сохранить допустимые добавления, не меняя завершённых результатов."""
        self._save()

    def seal(self):
        """Сохранить хеш полного выбора группы перед проверочными ошибками.

        Returns
        -------
        str
            Хеш selection_seal закреплённого выбора; повторный вызов проверяет
            сохранённый выбор.
        """
        self._active()
        if self.record.get("stage") in ("sealed", "scored"):
            self.require_sealed()
            return self.record["selection_seal"]
        if set(self.record["paths"]) != set(self._expected):
            raise CheckpointError("cannot seal before every planned path is present")
        for row in self.record["paths"].values():
            _path(row, terminal=True)
        self.record.update(stage="sealed", selection_seal=digest(_selection_payload(self.record)))
        self._save(sealing=True)
        return self.record["selection_seal"]

    def require_sealed(self):
        """Проверить закреплённый выбор группы.

        Returns
        -------
        str
            Хеш selection_seal, если стадия sealed или scored и запись совпадает
            с сохранённой.
        """
        self._active()
        if self.record.get("stage") not in ("sealed", "scored"):
            raise CheckpointError("truth/test scoring is unavailable before selection seal")
        _validate(self.record, self._bindings, self._expected)
        if not _same(self.record, self._committed):
            raise CheckpointError("only committed sealed content can expose scoring")
        return self.record["selection_seal"]

    def close(self):
        """Освободить файловую блокировку контрольной записи."""

        self._lease.close()

    def __enter__(self):
        """Проверить открытую контрольную запись и вернуть контекст.

        Returns
        -------
        Checkpoint
            Тот же экземпляр с действующей блокировкой.
        """

        self._active()
        return self

    def __exit__(self, *_):
        """Освободить блокировку контрольной записи при выходе из контекста."""

        self.close()


def validate_terminal_v2(record):
    """Проверить завершённость группы E05 и хеш сохранённого выбора.

    Сверяются состав последовательностей, сетка регуляризации и завершённость
    кандидатов. SHA-256 байтов файла проверяет read_terminal_v2.

    Parameters
    ----------
    record : dict
        Запись группы E05 со стадией scored.
    """
    if (type(record) is not dict or record.get("stage") != "scored"
            or type(record.get("bindings")) is not dict
            or type(record.get("paths")) is not dict
            or type(record.get("expected_paths")) is not list
            or not record["expected_paths"]
            or any(type(x) is not str or not x for x in record["expected_paths"])
            or len(set(record["expected_paths"])) != len(record["expected_paths"])
            or set(record["paths"]) != set(record["expected_paths"])
            or not _same(record.get("exponents"), list(ALPHA_EXPONENTS))):
        raise CheckpointError("v2 checkpoint is not a complete fixed-grid scored group")
    for pid, row in record["paths"].items():
        if pid.startswith("single/"):
            if type(row) is not dict or row.get("finalized") is not True:
                raise CheckpointError("v2 single-fit record is unfinished")
            if "candidate" in row:
                _candidate(row["candidate"])
        else:
            _path(row, terminal=True)
    if record.get("selection_seal") != digest(record["paths"]):
        raise CheckpointError("v2 selection seal mismatch")


def read_terminal_v2(path, *, expected_file_sha256, include_file_hash=False):
    """Прочитать завершённую группу E05 с заданным SHA-256 файла.

    Parameters
    ----------
    path : str or pathlib.Path
        Путь к исходному файлу группы.
    expected_file_sha256 : str
        Ожидаемый SHA-256 исходных байтов: 64 шестнадцатеричных символа в
        нижнем регистре.
    include_file_hash : bool, optional
        Вернуть вместе с записью хеш прочитанных байтов.

    Returns
    -------
    dict or tuple of (dict, str)
        Проверенная запись либо пара из записи и хеша файла. Привязки научных
        входов сверяются отдельно.
    """
    path = Path(path)
    if type(include_file_hash) is not bool:
        raise CheckpointError("include_file_hash must be boolean")
    if (type(expected_file_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is None):
        raise CheckpointError("expected_file_sha256 must be a lowercase SHA256 digest")
    raw = path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise CheckpointError("baseline file SHA256 differs from the admitted value")
    record = strict_json(raw)
    validate_terminal_v2(record)
    return (record, file_sha256) if include_file_hash else record
