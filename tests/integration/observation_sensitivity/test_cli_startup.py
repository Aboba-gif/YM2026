"""Проверки параллельного запуска CLI и границ блокировок с тестовой run_group."""
import os
from queue import Queue, Empty
import subprocess
import sys
from threading import Thread
import time

import pytest

from experiments.file_locks import FileLock, FileLockBusy


# Используются настоящие конфигурация, генераторы и допуск E05.
# Все тестовые кандидаты E05 отклонены искусственно.
SETUP = r'''
import sys
from pathlib import Path
from experiments.observation_sensitivity import cli
from experiments.observation_sensitivity import admission as mod
from experiments.observation_sensitivity.lifecycle import ALPHA_EXPONENTS
from experiments.observation_sensitivity.journal import Freeze, DirectJournal
from adrkit.config.validation import canonical_bytes, digest, strict_json

root = Path(sys.argv[1])
def put(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(record)+b'\n')
    return mod._sha(path.read_bytes())
registered = mod._registered_config_path()
spec = strict_json(registered.read_bytes())
spec['output'] = str(root/'source_recovery')
spec['protected_roots'] = []
for item in [spec['source_protocol'], *spec['input_files']]:
    item['path'] = str((registered.parent/item['path']).resolve())
config = root/'source_recovery.json'
code = {'adrkit/fixture.py': 'a' * 64}
bindings = dict(config_sha256=put(config, spec), code_files=code, code_sha256=digest(code),
    input_files={item['path']: item['sha256'] for item in [spec['source_protocol'], *spec['input_files']]},
    versions={'python': 'baseline-python', 'numpy': 'baseline-numpy', 'scipy': 'baseline-scipy'},
    threads={'OPENBLAS_NUM_THREADS': '8', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': None})
run = Path(spec['output'])/'run.json'
put(run, dict(schema=spec['schema'], configuration=spec, bindings=bindings, initial_state='estimating'))
protocol = mod.load_protocol(Path(spec['source_protocol']['path']), expected_sha256=spec['source_protocol']['sha256'])
sources = mod.make_sources(protocol, mass=spec['source_mass'])
for source in ('PG10','EC04'):
    for replicate in range(1,5):
        expected = mod._baseline_paths(spec, source, replicate)
        paths = {pid: dict(finalized=True) if pid.startswith('single/') else
            dict(finalized=True, alpha_reference=1., procedure_accepted=False,
                candidates={str(float(e)): dict(exponent=e, alpha=10.**e, accepted=False)
                    for e in ALPHA_EXPONENTS}) for pid in expected}
        put(run.parent/source/f'replicate_{replicate}.json', dict(stage='scored',
            exponents=list(ALPHA_EXPONENTS), expected_paths=expected, paths=paths,
            selection_seal=digest(paths), bindings=dict(bindings, source=source, replicate=replicate,
                source_record_sha256=mod.source_record_hash(sources[source]))))
config = root/'observation_sensitivity.json'
put(config, dict(schema=mod.CONFIG_SCHEMA, version=3, study=mod.STUDY,
    output=str(root/'observation_sensitivity'),
    design_sha256=digest(mod.design_record()), baseline=dict(kind='completed_run',
        config_path=str(root/'source_recovery.json'), run_manifest_path=str(run))))
admitted = mod.build_admission(config)
put(root/'admission.json', admitted.to_dict())
with Freeze(root/'observation_sensitivity', admitted) as journal:
    journal.freeze()
with DirectJournal(root/'observation_sensitivity', admitted) as journal:
    journal.start()
    journal.finish(dict(expected_fields=12, fields={name: {panel: dict(status='unavailable',
        test_only='No direct numerical calculation') for panel in ('D0','D1')} for name in sources}))
print('SETUP_COMPLETE', flush=True)
'''


# Код HOOK добавляет точки синхронизации при удержании блокировок.
# run_group заменена проверкой checkpoint; процедура допуска остаётся штатной.
HOOK = r'''
import os
if os.environ.get('E06_START_PROBE') == '1':
    import sys
    from experiments.observation_sensitivity import cli, journal, backend, driver
    original_read = journal._Journal._read
    waiting = False
    def read(self, name):
        global waiting
        if name == 'freeze.json' and not waiting:
            waiting = True
            assert self._read_only and self._lease.fd is not None
            assert not os.get_inheritable(self._lease.fd)
            print('JOURNAL_SHARED_HELD', flush=True)
            assert sys.stdin.readline().strip() == 'continue'
        return original_read(self, name)
    journal._Journal._read = read
    def forbidden(*args, **kwargs):
        raise AssertionError('A numerical solver must not run in this startup test')
    backend.ProductionBackend._solver = forbidden
    driver.fit = forbidden
    driver.final_certificate = forbidden
    def group(spec, paths, checkpoint, numerical_backend, panels, baseline, **kwargs):
        assert checkpoint._owner_pid == os.getpid() and checkpoint._lease.fd is not None
        assert not os.get_inheritable(checkpoint._lease.fd)
        assert checkpoint.record['stage'] == 'estimating'
        assert len(paths) == 24 and baseline is not None
        print('GROUP_READY', flush=True)
        assert sys.stdin.readline().strip() == 'finish'
        return checkpoint.record
    cli.run_group = group
'''


def _event(events, expected, count=2, timeout=90):
    deadline = time.monotonic() + timeout
    received, lines = set(), []
    while len(received) < count:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            pytest.fail(f"Timed out waiting for {expected}: {lines}")
        try:
            index, line = events.get(timeout=remaining)
        except Empty:
            pytest.fail(f"No subprocess event {expected}: {lines}")
        lines.append((index, line))
        if line == expected:
            received.add(index)
        if line == "<process-ended>" and index not in received:
            pytest.fail(f"CLI stopped before {expected}: {lines}")
    assert received == set(range(count)), lines


def test_two_real_cli_groups_share_startup_read_and_keep_independent_writers(tmp_path):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(HOOK, encoding="utf-8")
    root = tmp_path / "campaign"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E06_START_PROBE="0",
        PYTHONPATH=str(hooks)+os.pathsep+os.environ.get("PYTHONPATH", ""))
    setup = subprocess.run([sys.executable, "-B", "-c", SETUP, str(root)],
        env=env, capture_output=True, text=True, timeout=90)
    assert setup.returncode == 0 and setup.stdout.strip() == "SETUP_COMPLETE", (setup.stdout, setup.stderr)
    inputs = {p: p.read_bytes() for p in root.rglob("*.json")}
    env["E06_START_PROBE"] = "1"
    events, processes = Queue(), []
    def collect(index, process):
        for line in process.stdout:
            events.put((index, line.strip()))
        events.put((index, "<process-ended>"))
    try:
        for index in range(2):
            process = subprocess.Popen([sys.executable, "-B", "-m", "experiments.observation_sensitivity",
                "--config", str(root / "observation_sensitivity.json"), "group", "--source", "PG10",
                "--replicate", str(index+1)], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            processes.append(process)
            Thread(target=collect, args=(index, process), daemon=True).start()
        _event(events, "JOURNAL_SHARED_HELD")
        with pytest.raises(FileLockBusy):
            FileLock(root / "observation_sensitivity/.journal.lock")
        for process in processes:
            process.stdin.write("continue\n")
            process.stdin.flush()
        _event(events, "GROUP_READY")
        for rep in (1,2):
            with pytest.raises(FileLockBusy):
                FileLock(root / "observation_sensitivity/PG10" / f"replicate_{rep}.json.lock")
        for process in processes:
            process.stdin.write("finish\n")
            process.stdin.flush()
        for process in processes:
            process.wait(timeout=90)
            assert process.returncode == 0
        assert all(path.read_bytes() == raw for path, raw in inputs.items())
        with FileLock(root / "observation_sensitivity/.journal.lock"):
            pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=20)
