"""Чтение завершённых записей E06 под файловыми блокировками."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from adrkit.config.validation import canonical_bytes, strict_json
import adrkit
from experiments.file_locks import FileLock, FileLockBusy
from ..input_binding import validate_input_binding, scientific_input_roots


class SnapshotError(ValueError):
    """Расчёт занят, неполон, имеет псевдонимы путей или несогласованные записи."""


def _plain_path(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise SnapshotError("Filesystem aliases are not admitted")
    for node in (path, *path.parents):
        if node.is_symlink() or getattr(node, "is_junction", lambda: False)():
            raise SnapshotError("Filesystem aliases are not admitted")
    return path


class _ExistingLease(FileLock):
    """Удерживать исключительную неблокирующую блокировку существующего файла.

    Parameters
    ----------
    path : str or Path
        Существующий обычный файл блокировки без файловых псевдонимов; файл не
        создаётся.
    """

    def __init__(self, path):
        self.fd = None
        try:
            path = _plain_path(path)
            try:
                super().__init__(path, create=False)
            except ValueError as error:
                raise SnapshotError("Existing regular, unaliased lease required") from error
        except BaseException as error:
            self.close()
            if isinstance(error, (OSError, FileLockBusy)):
                raise SnapshotError("Existing lease unavailable; do not read campaign") from error
            raise


def _read(path):
    path = _plain_path(path)
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SnapshotError("Scientific input must be a regular, unaliased file")
            raw = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
        identities = [(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) for s in (before, after, current)]
        if any(s.st_nlink != 1 for s in (before, after, current)) or len(set(identities)) != 1:
            raise SnapshotError("Input changed or is hard-linked")
        value = strict_json(raw)
        if type(value) is not dict or raw != canonical_bytes(value) + b"\n":
            raise SnapshotError("Expected the exact canonical journal serialization")
        return value, hashlib.sha256(raw).hexdigest()
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, SnapshotError):
            raise
        raise SnapshotError(f"Cannot read committed input {path.name}") from error


def load_records(run_dir, *, expected_freeze_sha256):
    """Прочитать завершённые записи E06 под файловыми блокировками.

    Parameters
    ----------
    run_dir : str or Path
        Каталог расчёта с ``freeze.json``, ``direct.json`` и всеми
        объявленными завершёнными группами.
    expected_freeze_sha256 : str
        Независимо записанный SHA-256 канонического ``freeze.json`` с
        завершающим LF.

    Returns
    -------
    dict
        Отдельные JSON-объекты `freeze`, `direct`, `groups` и контрольные
        суммы `raw_file_sha256`. `direct` должен быть `completed`, все группы
        — `scored`.

    Notes
    -----
    На время чтения удерживаются существующие файлы блокировок;
    наличие файла незавершённой записи (`.json.pending`) препятствует
    чтению. Согласованность научных полей проверяет `summarize_records`.
    """
    if type(expected_freeze_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_freeze_sha256):
        raise SnapshotError("An independently recorded raw freeze SHA256 is required")
    run = _plain_path(run_dir)
    files = {"freeze": run / "freeze.json", "direct": run / "direct.json"}
    with ExitStack() as stack:
        stack.enter_context(_ExistingLease(run / ".journal.lock"))
        try:
            for path in files.values():
                if _plain_path(path.with_suffix(".json.pending")).exists():
                    raise SnapshotError("Unresolved pending write; no summary")
            freeze, frozen_sha = _read(files["freeze"])
            if frozen_sha != expected_freeze_sha256:
                raise SnapshotError("Freeze differs from independently recorded pin")
            admitted = freeze.get("admission", {})
            if admitted.get("output") != str(run):
                raise SnapshotError("Freeze names a different result directory")
            validate_input_binding(admitted)
            group_ids = tuple(admitted["expected_paths"])
            for key in group_ids:
                source, replicate = key.split("/r")
                target = run / source / f"replicate_{replicate}.json"
                files[key] = target
                stack.enter_context(_ExistingLease(target.with_suffix(".json.lock")))
                if _plain_path(target.with_suffix(".json.pending")).exists():
                    raise SnapshotError("Unresolved pending group write; no summary")
            records, digests = {"freeze": freeze}, {"freeze": frozen_sha}
            for key, path in files.items():
                if key != "freeze":
                    records[key], digests[key] = _read(path)
            if records["direct"].get("status") != "completed":
                raise SnapshotError("Direct journal is not terminal completed")
            groups = {key: records[key] for key in group_ids}
            if any(g.get("stage") != "scored" for g in groups.values()):
                raise SnapshotError("All admitted groups must be scored")
            return dict(freeze=freeze, direct=records["direct"], groups=groups,
                        raw_file_sha256=digests)
        except SnapshotError:
            raise
        except Exception as error:
            raise SnapshotError("Invalid or incomplete result records") from error


def add_read_arguments(parser):
    """Добавить обязательные аргументы чтения завершённого расчёта.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Объект разбора аргументов команды анализа.
    """
    parser.add_argument("--run", required=True, help="Каталог завершённого E06")
    parser.add_argument("--freeze-sha256", required=True, help="SHA-256 байтов freeze.json, отдельно сохранённый после закрепления")


def records_from_arguments(args):
    """Прочитать записи расчёта, заданного аргументами CLI.

    Parameters
    ----------
    args : argparse.Namespace
        Аргументы с полями `run` и `freeze_sha256`.

    Returns
    -------
    dict
        Результат `load_records` с `freeze`, `direct`, `groups` и
        `raw_file_sha256`.
    """
    return load_records(args.run, expected_freeze_sha256=args.freeze_sha256)


def input_root(args):
    """Определить абсолютный каталог входных записей CLI.

    Parameters
    ----------
    args : argparse.Namespace
        Аргументы с полем `run`.

    Returns
    -------
    Path
        Абсолютный путь `run` без файловых псевдонимов.
    """
    return _plain_path(args.run)


def export_destination(output, *, protected_root, additional_roots=()):
    """Проверить новый путь вывода вне входов и загруженных исходников.

    Parameters
    ----------
    output : str or Path
        Ещё не существующий файл или каталог вывода.
    protected_root : str or Path
        Корень входных результатов, с которым вывод не должен пересекаться.
    additional_roots : iterable of str or Path, optional
        Дополнительные защищённые корни.

    Returns
    -------
    Path
        Абсолютный допустимый путь. Родительские каталоги здесь не создаются.

    Raises
    ------
    SnapshotError
        Путь существует, содержит файловый псевдоним или пересекает защищённый
        корень.
    """
    destination = _plain_path(Path(os.path.abspath(output)))
    roots = (Path(protected_root).resolve(), Path(__file__).resolve().parents[3],
             Path(adrkit.__file__).resolve().parent, *(Path(p).resolve() for p in additional_roots))
    if destination.exists() or any(destination.is_relative_to(root) or root.is_relative_to(destination) for root in roots):
        raise SnapshotError("Choose a new output outside the inputs and installed code")
    return destination


def main(argv=None):
    """Прочитать завершённый расчёт E06 и записать сводку парных эффектов.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы без имени программы; None использует ``sys.argv[1:]``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    add_read_arguments(parser)
    parser.add_argument("--output", required=True, help="Новый файл JSON сводки завершённого расчёта")
    args = parser.parse_args(argv)
    protected = input_root(args)
    records = records_from_arguments(args)
    destination = export_destination(args.output, protected_root=protected,
        additional_roots=scientific_input_roots(records["freeze"]["admission"]))
    from . import paired_effects as analyzer
    result = analyzer.summarize_records(records["freeze"], records["direct"], records["groups"],
                                       expected_freeze_sha256=args.freeze_sha256)
    result["raw_input_file_sha256"] = records["raw_file_sha256"]
    result["snapshot_reader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["analyzer_sha256"] = hashlib.sha256(Path(analyzer.__file__).read_bytes()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(destination), "input_files": len(records["raw_file_sha256"])}))


if __name__ == "__main__":
    main()
