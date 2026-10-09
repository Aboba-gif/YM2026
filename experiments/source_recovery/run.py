"""Запуск опыта восстановления источника из файловой конфигурации."""
from __future__ import annotations

import json
import os
import platform
from hashlib import sha256
from pathlib import Path
import numpy as np
import scipy
import adrkit
from experiments.file_locks import FileLock
from experiments.source_comparison.config import load_protocol
from .checkpoints import canonical_hash, file_hash, atomic_json, Checkpoint
from .config import SCHEMA, validate_config
from .driver import _conditions, run_group


def source_record_hash(source):
    """Вычислить SHA-256 записи зарегистрированного источника.

    Parameters
    ----------
    source : ScaledSource
        Масштабированный аналитический источник.

    Returns
    -------
    digest : str
        Шестнадцатеричный хеш после преобразования записи к стандартным
        типам JSON и канонизации.
    """
    from .sources import source_record
    record = json.loads(json.dumps(source_record(source), allow_nan=False))
    return canonical_hash(record)


def run_experiment(config_path, *, source_id=None, replicate=None):
    """Выполнить или продолжить опыт восстановления источника.

    Входы и привязки проверяются до численной работы. Записи завершённых
    кандидатов используются при продолжении. Инициализация общего манифеста
    выполняется под короткой блокировкой; запись каждой группы выполняет один процесс.

    Parameters
    ----------
    config_path : str or path-like
        JSON-конфигурация с явно закреплёнными научными условиями.
    source_id : str or None, optional
        Один зарегистрированный источник; None запускает весь список.
    replicate : int or None, optional
        Одна зарегистрированная реализация; None запускает весь список.

    Returns
    -------
    output : Path
        Каталог описания расчёта и файлов состояния по источникам и
        реализациям.
    """

    from .backend import ProductionBackend
    from .sources import make_sources
    path = Path(config_path).resolve()
    config_bytes = path.read_bytes()
    spec = json.loads(config_bytes.decode("utf-8"))
    config_sha256 = sha256(config_bytes).hexdigest()
    validate_config(spec)
    reference = spec["source_protocol"]
    protocol_path = (path.parent/reference["path"]).resolve()
    protocol = load_protocol(protocol_path,expected_sha256=reference["sha256"])
    inputs = {str(protocol_path):protocol.full_sha256}
    for item in spec["input_files"]:
        target = (path.parent/item["path"]).resolve()
        if target == path:
            actual = config_sha256
        elif target == protocol_path:
            actual = protocol.full_sha256
        else:
            actual = file_hash(target)
        if actual != item["sha256"]:
            raise ValueError(f"Input hash changed: {target.name}")
        inputs[str(target)] = actual
    experiment_root = Path(__file__).resolve().parents[1]
    code_roots = {"adrkit": Path(adrkit.__file__).resolve().parent,
                  "experiments/source_comparison": experiment_root/"source_comparison",
                  "experiments/source_recovery": experiment_root/"source_recovery"}
    code = {name+"/"+p.relative_to(root).as_posix(): file_hash(p)
            for name, root in code_roots.items()
            for p in sorted(root.rglob("*.py")) if "analysis" not in p.relative_to(root).parts}
    code["experiments/file_locks.py"] = file_hash(experiment_root/"file_locks.py")
    bindings = dict(config_sha256=config_sha256,input_files=inputs,
        code_sha256=canonical_hash(code),code_files=code,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        threads={key:os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")})
    sources = make_sources(protocol,mass=spec["source_mass"])
    # Все генераторы проверяются до записи описания расчёта и начала численной работы, также при
    # выборе одного источника.
    source_hashes = {name:source_record_hash(source) for name,source in sources.items()}
    names = spec["sources"] if source_id is None else [source_id]
    reps = spec["replicates"] if replicate is None else [replicate]
    if not set(names) <= set(spec["sources"]) or not set(reps) <= set(spec["replicates"]):
        raise ValueError("Requested source/replicate not in frozen manifest")
    output = (path.parent/spec["output"]).resolve()
    project = Path(__file__).resolve().parents[2]
    protected = [project, Path(adrkit.__file__).resolve().parent,
                 *(path.parent/entry for entry in spec.get("protected_roots", []))]
    if any(output == root.resolve() or output.is_relative_to(root.resolve())
           or root.resolve().is_relative_to(output) for root in protected):
        raise ValueError("New experiment output must not overlap source or protected input directories")
    if any(output == target or target.is_relative_to(output) for target in (path, *map(Path, inputs))):
        raise ValueError("New experiment output must not contain scientific input files")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"run.json"
    with FileLock(output/".run.lock", blocking=True):
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old["bindings"] != bindings:
                raise ValueError("Run manifest code/config/inputs changed; use a new output")
        else:
            atomic_json(manifest,dict(schema=SCHEMA,bindings=bindings,configuration=spec,
                initial_state="frozen configuration admitted; checkpoints own execution state"))
    for name in names:
        for r in reps:
            conditions = _conditions(spec,name,r)
            if not conditions:
                continue
            expected = [c["id"]+"/"+arm for c in conditions for arm in c["penalties"]]
            expected += ["single/"+s["id"] for s in spec["single_fits"] if name in s["sources"] and r in s["replicates"]]
            bound = dict(bindings,source=name,replicate=r,source_record_sha256=source_hashes[name])
            checkpoint_path = output/name/f"replicate_{r}.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(checkpoint_path.with_suffix(".json.lock")):
                checkpoint = Checkpoint(checkpoint_path,bound,expected)
                backend = ProductionBackend(spec,sources[name])
                run_group(spec,name,r,checkpoint,backend)
                del backend
    return output
