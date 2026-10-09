"""Проверки владельца E05 CLI; run_group заменена записью тестового состояния."""
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Thread

import pytest

from experiments.file_locks import FileLock, FileLockBusy
from experiments.source_recovery.config import template_config


HOOK = r'''
import os
if os.environ.get('E05_OWNER_TEST') == '1':
    import sys
    from experiments.file_locks import FileLockBusy
    from experiments.source_recovery import run, backend
    def receive():
        command = sys.stdin.readline().strip()
        if not command or command == 'stop':
            raise SystemExit(0)
        return command
    OriginalLock = run.FileLock
    class ObservedLock(OriginalLock):
        def __init__(self, path, **options):
            gated = path.name == '.run.lock' and os.environ.get('E05_MANIFEST_GATE') == '1'
            if gated:
                print('MANIFEST_REQUEST', flush=True)
                assert receive() == 'request'
            super().__init__(path, **options)
            assert not os.get_inheritable(self.fd)
            if gated:
                print('MANIFEST_HELD', flush=True)
                try:
                    assert receive() == 'publish'
                except BaseException:
                    self.close()
                    raise
    run.FileLock = ObservedLock
    OriginalCheckpoint = run.Checkpoint
    OriginalBackend = backend.ProductionBackend
    def checkpoint(*args, **kwargs):
        print('CHECKPOINT_READ', flush=True)
        return OriginalCheckpoint(*args, **kwargs)
    def numerical_backend(*args, **kwargs):
        print('BACKEND_INIT', flush=True)
        return OriginalBackend(*args, **kwargs)
    run.Checkpoint = checkpoint
    backend.ProductionBackend = numerical_backend
    def group(spec, source, replicate, checkpoint, backend):
        assert source == 'PG10' and checkpoint.record['bindings']['replicate'] == replicate
        lock = checkpoint.path.with_suffix('.json.lock')
        try:
            with OriginalLock(lock):
                pass
        except FileLockBusy:
            pass
        else:
            raise AssertionError('Checkpoint was read without a group owner')
        key = checkpoint.record['expected_paths'][0]
        if key in checkpoint.record['paths']:
            assert checkpoint.record['paths'][key] == {'ownership_test_saved': True}
            print('RESTORED', flush=True)
        print('GROUP_READY', flush=True)
        command = receive()
        if command == 'save':
            checkpoint.record['paths'][key] = {'ownership_test_saved': True}
            checkpoint.save()
            print('SAVED', flush=True)
            command = receive()
        if command == 'abrupt':
            os._exit(17)
        if command == 'fail':
            raise RuntimeError('Injected caller failure after committed test state')
        assert command == 'finish'
        return checkpoint.record
    run.run_group = group
'''


@pytest.fixture
def campaign(tmp_path, project_root):
    spec = template_config(100.)
    spec.update(frozen=True, sources=['PG10'], replicates=[1, 2], output=str(tmp_path / 'results'))
    spec['source_protocol']['path'] = str(project_root / 'experiments/source_comparison/configs/protocol.json')
    spec['conditions'] = [dict(id='main', sources=['PG10'], replicates=[1, 2],
        weight='W01', penalties=['L2', 'H1'], grid='G0', nodes=73, noise='corr',
        tau_hours=.25, truth_gamma=1/72, inverse_gamma=1/72, availability='full',
        mask='none', temporal_H='snapshot', relocation_km=0.)]
    config = tmp_path / 'experiment.json'
    config.write_text(json.dumps(spec), encoding='utf-8')
    hooks = tmp_path / 'hooks'
    hooks.mkdir()
    (hooks / 'sitecustomize.py').write_text(HOOK, encoding='utf-8')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', E05_OWNER_TEST='1',
        PYTHONPATH=str(hooks) + os.pathsep + str(project_root))
    return config, Path(spec['output']), env


def start(config, env, replicate, *, manifest_gate=False):
    env = dict(env, E05_MANIFEST_GATE='1' if manifest_gate else '0')
    process = subprocess.Popen([sys.executable, '-B', '-m', 'experiments.source_recovery',
        '--config', str(config), '--source', 'PG10', '--replicate', str(replicate)],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    events = Queue()
    def collect():
        for line in process.stdout:
            events.put(line.strip())
        events.put('<process-ended>')
    collector = Thread(target=collect, daemon=True)
    try:
        collector.start()
    except BaseException:
        stop([(process, events, collector)])
        raise
    return process, events, collector


def expect(child, expected):
    lines = []
    while True:
        line = child[1].get(timeout=60)
        lines.append(line)
        assert line != '<process-ended>', lines
        if line == expected:
            return lines


def send(child, command):
    child[0].stdin.write(command + '\n')
    child[0].stdin.flush()


def stop(children):
    # Дать команду всем дочерним процессам до ожидания: один может ждать блокировку другого.
    for process, _, _ in children:
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.write('stop\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
    for process, _, collector in children:
        process.wait(timeout=60)
        if collector.ident is not None:
            collector.join(timeout=60)
            assert not collector.is_alive(), 'Native child did not close its stdout pipe'
        elif process.stdout is not None:
            process.stdout.close()


def test_two_fresh_cli_groups_initialize_one_manifest_and_exclude_same_group(campaign):
    config, output, env = campaign
    children = []
    try:
        first = start(config, env, 1, manifest_gate=True)
        children.append(first)
        second = start(config, env, 2, manifest_gate=True)
        children.append(second)
        expect(first, 'MANIFEST_REQUEST')
        expect(second, 'MANIFEST_REQUEST')
        assert not (output / 'run.json').exists()
        send(first, 'request')
        expect(first, 'MANIFEST_HELD')
        send(second, 'request')
        with pytest.raises(FileLockBusy):
            FileLock(output / '.run.lock')
        send(first, 'publish')
        expect(first, 'GROUP_READY')
        expect(second, 'MANIFEST_HELD')
        manifest = (output / 'run.json').read_bytes()
        send(second, 'publish')
        expect(second, 'GROUP_READY')
        assert first[0].poll() is None and second[0].poll() is None
        checkpoints = [output / 'PG10' / f'replicate_{r}.json' for r in (1, 2)]
        before = {path: path.read_bytes() for path in checkpoints}
        for path in checkpoints:
            with pytest.raises(FileLockBusy):
                FileLock(path.with_suffix('.json.lock'))
        occupied = start(config, env, 1)
        children.append(occupied)
        assert occupied[0].wait(timeout=60) != 0
        output_lines = []
        while True:
            line = occupied[1].get(timeout=60)
            if line == '<process-ended>':
                break
            output_lines.append(line)
        assert any('FileLockBusy' in line for line in output_lines)
        assert not {'CHECKPOINT_READ', 'BACKEND_INIT', 'GROUP_READY'} & set(output_lines)
        assert before == {path: path.read_bytes() for path in checkpoints}
        assert (output / 'run.json').read_bytes() == manifest
        for child in (first, second):
            send(child, 'finish')
        for process, _, _ in (first, second):
            assert process.wait(timeout=60) == 0
        assert (output / 'run.json').read_bytes() == manifest
        assert before == {path: path.read_bytes() for path in checkpoints}
        inventory = json.loads(manifest)['bindings']['code_files']
        assert 'experiments/file_locks.py' in inventory
        with FileLock(output / '.run.lock'):
            pass
        for path in checkpoints:
            with FileLock(path.with_suffix('.json.lock')):
                pass
    finally:
        stop(children)


@pytest.mark.parametrize('interruption', ['abrupt', 'fail'])
def test_committed_group_survives_owner_exit_and_restarts_while_neighbor_runs(campaign, interruption):
    config, output, env = campaign
    children = []
    try:
        owner = start(config, env, 1)
        children.append(owner)
        expect(owner, 'GROUP_READY')
        send(owner, 'save')
        expect(owner, 'SAVED')
        path = output / 'PG10/replicate_1.json'
        committed = path.read_bytes()
        manifest = (output / 'run.json').read_bytes()
        neighbor = start(config, env, 2)
        children.append(neighbor)
        expect(neighbor, 'GROUP_READY')
        send(owner, interruption)
        assert owner[0].wait(timeout=60) == (17 if interruption == 'abrupt' else 1)
        assert neighbor[0].poll() is None
        assert path.read_bytes() == committed
        resumed = start(config, env, 1)
        children.append(resumed)
        assert 'RESTORED' in expect(resumed, 'GROUP_READY')
        assert path.read_bytes() == committed
        assert (output / 'run.json').read_bytes() == manifest
        send(resumed, 'finish')
        send(neighbor, 'finish')
        assert resumed[0].wait(timeout=60) == neighbor[0].wait(timeout=60) == 0
        assert path.read_bytes() == committed
        assert (output / 'run.json').read_bytes() == manifest
    finally:
        stop(children)
