"""Инициализация проверки файлового хранилища в процессах, запущенных через ``spawn``; ADR не решается."""
import json
from multiprocessing import get_start_method
import os
from time import monotonic, sleep

from experiments.file_locks import FileLock, FileLockBusy
from experiments.source_comparison import run


def install_compute_stub():
    """Заменить научный расчёт тестовой функцией в каждом дочернем процессе."""
    run._compute_source = compute_stub


def require_busy(path):
    """Проверить, что занятую блокировку нельзя получить через отдельный дескриптор."""
    try:
        with FileLock(path, create=False):
            raise AssertionError(f'Expected an active ownership lease: {path.name}')
    except FileLockBusy:
        pass


def compute_stub(settings, protocol, output, source_id, bindings):
    """Синхронизировать два дочерних процесса и записать искусственные метаданные источников."""
    require_busy(output/'.run.lock')
    require_busy(output/(source_id+'.json.lock'))
    assert get_start_method() == 'spawn'
    assert protocol.full_sha256 == bindings['protocol_sha256']
    assert len(settings['sources']) == 2
    pid = os.getpid()
    ready = output/(source_id+'.spawn-ready')
    run.save_json(ready, dict(source=source_id, pid=pid))
    markers = [output/(source+'.spawn-ready') for source in settings['sources']]
    deadline = monotonic()+20
    while not all(marker.is_file() for marker in markers):
        if monotonic() >= deadline:
            raise TimeoutError('Two spawned storage workers did not rendezvous')
        sleep(.01)
    pids = {json.loads(marker.read_bytes())['pid'] for marker in markers}
    if len(pids) != 2:
        raise AssertionError('Both sources must be active in different spawned processes')
    run.save_json(output/(source_id+'.json'), dict(source=source_id,
        bindings=bindings, pid=pid, start_method=get_start_method(),
        test_only='Artificial storage probe; no scientific computation'))
    return source_id, 'test_only'
