"""Фиксация постановки E06 и атомарное сохранение прямого расчёта."""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
import os
from pathlib import Path
from uuid import UUID, uuid4

from adrkit.config.validation import canonical_bytes, digest, strict_json
from .admission import Admission, AdmissionError, _plain_path
from .input_binding import validate_input_binding
from .lifecycle import (
    _Lease, _replace_pending, CheckpointError, PendingRecoveryRequired,
    PersistenceError,
)


class JournalError(CheckpointError):
    """Неверная привязка, запись, путь или переход состояния журнала."""


class UnresolvedStart(JournalError):
    """Начало записано, завершение неизвестно; автоматический повтор исключён."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _time(value):
    try:
        parsed = datetime.fromisoformat(value) if type(value) is str else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise JournalError("Journal timestamps must specify UTC")


def _same(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _sealed(record):
    return dict(record, content_sha256=digest(record))


def _check_digest(record):
    if (type(record) is not dict or record.get("content_sha256") !=
            digest({key: value for key, value in record.items() if key != "content_sha256"})):
        raise JournalError("Journal content digest mismatch")


class _Journal:
    """Общая атомарная запись журнала E06 с файловой блокировкой.

    Экземпляр поддерживает контекстный менеджер. После ошибки сохранения его
    следует закрыть, сохранив подготовленный файл для явного восстановления.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Абсолютный каталог, совпадающий с выходным каталогом допуска.
    admission : Admission
        Проверенный допуск E06 версии 3 или 4.
    read_only : bool, optional
        Открыть существующий каталог и файл блокировки без записи; блокировка
        совместная.
    replace_attempts : int, optional
        Положительное число попыток атомарной замены файла.
    retry_delay : float, optional
        Начальная неотрицательная задержка повторной замены в секундах.
    """
    _filename = None

    def __init__(self, output_dir, admission, *, read_only=False, replace_attempts=5, retry_delay=.05):
        if type(read_only) is not bool:
            raise JournalError("read_only must be boolean")
        if not isinstance(admission, Admission):
            raise TypeError("An explicit Admission snapshot is required")
        declared = admission.to_dict()
        if (declared.get("schema") != "ym2026.observation_sensitivity.admission"
                or type(declared.get("version")) is not int or declared["version"] not in (3, 4)
                or declared.get("status") != "prepared_not_frozen"
                or digest(declared) != admission.sha256):
            raise JournalError("Only canonical version-3 or version-4 scientific admissions are writable")
        try:
            validate_input_binding(declared)
        except (ValueError, TypeError, KeyError) as error:
            raise JournalError(f"Invalid scientific input binding: {error}") from error
        self._version = declared["version"]
        if (type(replace_attempts) is not int or replace_attempts < 1
                or type(retry_delay) not in (int, float)
                or not isfinite(retry_delay) or retry_delay < 0):
            raise JournalError("Positive retry count and finite nonnegative delay required")
        try:
            self.output_dir = _plain_path(output_dir, label="journal output")
        except AdmissionError as error:
            raise JournalError("Journal output must not be a filesystem alias") from error
        target = declared.get("output")
        if (type(target) is not str or not Path(target).is_absolute()
                or self.output_dir != Path(target).resolve()):
            raise JournalError("Caller output directory must equal the admitted E06 output")
        self._admission = canonical_bytes(declared)
        self._admission_sha = admission.sha256
        self._attempts, self._delay = replace_attempts, retry_delay
        self._owner_pid, self._poisoned, self._lease = os.getpid(), False, None
        self._record = None
        self._read_only = read_only
        if read_only:
            if not self.output_dir.is_dir():
                raise JournalError("Read-only journal directory does not exist")
            try:
                self._lease = _Lease(self._path(".journal.lock"), shared=True, create=False)
            except FileNotFoundError as error:
                raise JournalError("Read-only journal lock file does not exist") from error
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._lease = _Lease(self._path(".journal.lock"))

    def _path(self, name):
        path = self.output_dir/name
        if (path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                or path.resolve() != path):
            raise JournalError("Journal files must not be filesystem aliases")
        return path

    def _active(self):
        if (self._owner_pid != os.getpid() or self._lease is None
                or self._lease.fd is None or self._poisoned):
            raise PersistenceError("Journal is closed, belongs to another process, or had a failed write; reopen and inspect records")

    def _read(self, name):
        path = self._path(name)
        if not path.exists():
            return None
        try:
            record = strict_json(path.read_bytes())
            _check_digest(record)
            return record
        except (OSError, ValueError, TypeError) as error:
            raise JournalError(f"Cannot validate journal record {name}: {error}") from error

    def _writable(self):
        self._active()
        if self._read_only:
            raise JournalError("Read-only journal cannot write or commit records")

    def _no_pending(self, name):
        if self._path(name+".pending").exists():
            raise PendingRecoveryRequired("Prepared journal write exists; inspect and explicitly commit_pending()")

    def _write(self, name, record):
        self._writable()
        self._no_pending(name)
        raw = canonical_bytes(record)+b"\n"
        target, pending = self._path(name), self._path(name+".pending")
        try:
            with pending.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            _replace_pending(pending, target, self._attempts, self._delay)
        except OSError as error:
            self._poisoned = True
            raise PersistenceError("Journal save failed; preserve pending evidence, never recalculate") from error
        self._record = canonical_bytes(record)
        return self.record

    def _commit(self, name, record):
        self._writable()
        try:
            _replace_pending(self._path(name+".pending"), self._path(name), self._attempts, self._delay)
        except OSError as error:
            self._poisoned = True
            raise PersistenceError("Prepared journal commit failed; no numerical retry") from error
        self._record = canonical_bytes(record)
        return self.record

    @property
    def record(self):
        """Вернуть отдельную копию записи журнала.

        Returns
        -------
        dict or None
            Сохранённая запись либо None. Изменение результата не меняет журнал и
            файл.
        """
        return None if self._record is None else strict_json(self._record)

    @property
    def pending_exists(self):
        """Проверить наличие подготовленной записи.

        Returns
        -------
        bool
            True, если файл pending существует; запись не завершается.
        """
        self._active()
        return self._path(self._filename + ".pending").exists()

    def close(self):
        """Освободить файловую блокировку журнала."""

        if self._lease is not None:
            self._lease.close()

    def __enter__(self):
        """Проверить открытый журнал и вернуть контекст.

        Returns
        -------
        _Journal
            Тот же открытый экземпляр журнала.
        """

        self._active()
        return self

    def __exit__(self, *_):
        """Освободить блокировку журнала при выходе из контекста."""

        self.close()

    def _freeze_record(self, record):
        expected = {"schema", "version", "admission", "admission_sha256", "frozen_at", "content_sha256"}
        if (type(record) is not dict or set(record) != expected
                or record["schema"] != "ym2026.observation_sensitivity.freeze"
                or type(record["version"]) is not int or record["version"] != self._version
                or record["admission_sha256"] != self._admission_sha
                or canonical_bytes(record["admission"]) != self._admission):
            raise JournalError("Freeze differs from the current admission; resume forbidden")
        _check_digest(record)
        _time(record["frozen_at"])
        return record


class Freeze(_Journal):
    """Журнал закрепления проверенных входов E06.

    Повторная фиксация одинакового допуска сохраняет исходное время. При
    открытии уже существующая запись проверяется; подготовленная запись
    завершается только через commit_pending.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Абсолютный каталог, совпадающий с выходным каталогом допуска.
    admission : Admission
        Проверенный допуск E06 версии 3 или 4.
    read_only : bool, optional
        Открыть существующий каталог и файл блокировки без записи; блокировка
        совместная.
    replace_attempts : int, optional
        Положительное число попыток атомарной замены файла.
    retry_delay : float, optional
        Начальная неотрицательная задержка повторной замены в секундах.
    """
    _filename = "freeze.json"

    def __init__(self, output_dir, admission, **options):
        super().__init__(output_dir, admission, **options)
        try:
            existing = self._read("freeze.json")
            if existing is not None:
                self._record = canonical_bytes(self._freeze_record(existing))
        except BaseException:
            self.close()
            raise

    def freeze(self):
        """Сохранить допуск либо вернуть совпадающую закреплённую запись.

        Returns
        -------
        dict
            Закреплённый допуск с исходным временем и контрольной суммой.
        """
        self._writable()
        self._no_pending("freeze.json")
        if self._record is not None:
            return self.require_frozen()
        if self._read("freeze.json") is not None:
            raise JournalError("Frozen record appeared during the lease")
        record = _sealed(dict(schema="ym2026.observation_sensitivity.freeze", version=self._version,
            admission=strict_json(self._admission), admission_sha256=self._admission_sha, frozen_at=_now()))
        return self._write("freeze.json", record)

    def require_frozen(self):
        """Проверить совпадение закреплённого допуска с текущим снимком.

        Returns
        -------
        dict
            Проверенная запись freeze.json; новый словарь.
        """
        self._active()
        self._no_pending("freeze.json")
        record = self._freeze_record(self._read("freeze.json"))
        if self._record is not None and not _same(record, self.record):
            raise JournalError("Frozen record changed during the lease")
        return record

    def commit_pending(self):
        """Проверить и завершить подготовленную запись допуска.

        Returns
        -------
        dict
            Запись freeze.json после завершения сохранения; новый словарь.
        """
        self._writable()
        pending = self._freeze_record(self._read("freeze.json.pending"))
        existing = self._read("freeze.json")
        if existing is not None and not _same(self._freeze_record(existing), pending):
            raise JournalError("Prepared freeze cannot replace a different immutable freeze")
        return self._commit("freeze.json", pending)


_START_KEYS = {"schema", "version", "admission_sha256", "freeze_sha256",
               "run_id", "started_at"}


class DirectJournal(_Journal):
    """Журнал одного прямого запуска E06.

    Требуется совпадающий закреплённый допуск. Повторное открытие начатой
    записи не запускает расчёт; результат может завершить только контекст,
    записавший начало.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Абсолютный каталог, совпадающий с выходным каталогом допуска.
    admission : Admission
        Проверенный допуск E06 версии 3 или 4.
    read_only : bool, optional
        Открыть существующий каталог и файл блокировки без записи; блокировка
        совместная.
    replace_attempts : int, optional
        Положительное число попыток атомарной замены файла.
    retry_delay : float, optional
        Начальная неотрицательная задержка повторной замены в секундах.
    """
    _filename = "direct.json"

    def __init__(self, output_dir, admission, **options):
        super().__init__(output_dir, admission, **options)
        self._owned_start = None
        try:
            self._no_pending("freeze.json")
            frozen = self._freeze_record(self._read("freeze.json"))
            self._freeze_sha = frozen["content_sha256"]
            existing = self._read("direct.json")
            if existing is not None:
                self._record = canonical_bytes(self._validate_record(existing))
        except BaseException:
            self.close()
            raise

    def _validate_record(self, record):
        if type(record) is not dict:
            raise JournalError("No prepared direct record")
        status = record.get("status")
        expected = _START_KEYS | {"status", "content_sha256"}
        if status in ("completed", "partial"):
            expected |= {"finished_at", "result"}
        if (set(record) != expected or status not in ("started", "completed", "partial")
                or record["schema"] != "ym2026.observation_sensitivity.direct_journal"
                or type(record["version"]) is not int or record["version"] != self._version
                or record["admission_sha256"] != self._admission_sha
                or record["freeze_sha256"] != self._freeze_sha):
            raise JournalError("Invalid direct journal structure or freeze/admission binding")
        try:
            if type(record["run_id"]) is not str or str(UUID(record["run_id"])) != record["run_id"]:
                raise ValueError("noncanonical UUID")
        except (ValueError, TypeError, AttributeError) as error:
            raise JournalError("Invalid direct invocation ID") from error
        _time(record["started_at"])
        if status != "started":
            _time(record["finished_at"])
            if type(record["result"]) is not dict or not record["result"]:
                raise JournalError("Terminal/partial evidence must be a nonempty JSON object")
        _check_digest(record)
        return record

    def _check_freeze(self):
        self._no_pending("freeze.json")
        frozen = self._freeze_record(self._read("freeze.json"))
        if frozen["content_sha256"] != self._freeze_sha:
            raise JournalError("Freeze changed during direct invocation")

    def _current(self):
        self._active()
        self._no_pending("direct.json")
        self._check_freeze()
        current = self._read("direct.json")
        if current is not None:
            self._validate_record(current)
        if not _same(current, self.record):
            raise JournalError("Direct journal changed during the lease")
        return current

    def start(self):
        """Сохранить начало единственного прямого запуска в этом контексте.

        Returns
        -------
        dict
            Запись со статусом started и новым run_id; отдельная JSON-копия.
        """
        self._writable()
        current = self._current()
        if current is not None:
            if current["status"] == "started":
                raise UnresolvedStart("Direct run already started; no automatic rerun or guessed recovery")
            raise JournalError("Direct run already has terminal/partial evidence; no rerun")
        record = _sealed(dict(schema="ym2026.observation_sensitivity.direct_journal", version=self._version,
            admission_sha256=self._admission_sha, freeze_sha256=self._freeze_sha,
            run_id=str(uuid4()), started_at=_now(), status="started"))
        result = self._write("direct.json", record)
        self._owned_start = record["run_id"]
        return result

    def finish(self, result, *, partial=False):
        """Сохранить результат прямого запуска, начатого в этом контексте.

        Parameters
        ----------
        result : dict
            Непустая JSON-запись результата прямой серии.
        partial : bool, optional
            Сохранить статус partial вместо completed.

        Returns
        -------
        dict
            Отдельная копия завершённой записи с результатом и временем
            окончания.
        """
        self._writable()
        current = self._current()
        if (current is None or current["status"] != "started"
                or current["run_id"] != self._owned_start):
            raise UnresolvedStart("Only this context's committed start may publish its result")
        if type(partial) is not bool or type(result) is not dict or not result:
            raise JournalError("Explicit boolean partial and nonempty JSON result required")
        canonical_bytes(result)
        record = {key: current[key] for key in _START_KEYS}
        record.update(status="partial" if partial else "completed", finished_at=_now(), result=result)
        return self._write("direct.json", _sealed(record))

    def require_terminal(self):
        """Вернуть завершённую или частичную запись прямого запуска.

        Returns
        -------
        dict
            Проверенная запись со статусом completed или partial; незавершённое
            начало отклоняется.
        """
        current = self._current()
        if current is None or current["status"] == "started":
            raise UnresolvedStart("No committed terminal/partial direct result")
        return current

    def commit_pending(self):
        """Проверить и завершить подготовленную запись прямого запуска.

        Returns
        -------
        dict
            Зафиксированная запись; вычисления не запускаются, право завершать
            чужое начало не предоставляется.
        """
        self._writable()
        self._check_freeze()
        pending = self._validate_record(self._read("direct.json.pending"))
        current = self._read("direct.json")
        if current is None:
            if pending["status"] != "started":
                raise JournalError("Prepared result cannot be committed without its start marker")
        else:
            self._validate_record(current)
            if not _same(current, pending):
                if (current["status"] != "started" or pending["status"] == "started"
                        or not _same({key: current[key] for key in _START_KEYS},
                                     {key: pending[key] for key in _START_KEYS})):
                    raise JournalError("Prepared result does not extend the committed start")
        # Завершение записи прерванного начала не запускает численный расчёт.
        self._owned_start = None
        return self._commit("direct.json", pending)
