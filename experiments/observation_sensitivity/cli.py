"""Команды допуска, фиксации входов, прямой серии и групп восстановления E06."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adrkit.config.validation import digest
from experiments.source_comparison.config import load_protocol
from experiments.source_recovery.run import source_record_hash
from experiments.source_recovery.sources import make_sources
from .admission import build_admission, _plain_path
from .backend import ProductionBackend
from .data import PanelFactory
from .direct import collect_direct, DirectContractError
from .design import default_direct_spec
from .driver import run_group
from .journal import Freeze, DirectJournal
from .lifecycle import Checkpoint, CheckpointError
from .reuse import VerifiedBaseline


def _sources(admission):
    record, spec = admission.to_dict(), admission.spec
    protocol_path = _plain_path(record['source_protocol_path'], label='source protocol')
    protocol = load_protocol(protocol_path,
        expected_sha256=spec['source_protocol']['sha256'])
    sources = make_sources(protocol, mass=spec['source_mass'])
    if not set(record['sources']) <= set(sources) or any(
            source_record_hash(sources[name]) != value['sha256']
            for name, value in record['sources'].items()):
        raise CheckpointError('Source factory no longer matches the admitted generators')
    return {name: sources[name] for name in record['sources']}


def _require_freeze(admission):
    with Freeze(admission.to_dict()['output'], admission, read_only=True) as journal:
        return journal.require_frozen()


def _direct(admission, *, commit_pending=False):
    _require_freeze(admission)
    output = admission.to_dict()['output']
    with DirectJournal(output, admission) as journal:
        if commit_pending:
            # Эта операция не вызывает collect_direct, включая восстановление начатой записи.
            return journal.commit_pending()
        if journal.pending_exists:
            raise CheckpointError('Prepared direct bytes require explicit commit; no calculation is allowed')
        if journal.record is not None:
            if journal.record['status'] in ('completed', 'partial'):
                return journal.record
            raise CheckpointError('Direct start remains unresolved; automatic rerun is forbidden')
        sources = _sources(admission)
        journal.start()
        try:
            result = collect_direct(admission.spec, sources, direct=admission.to_dict().get('direct'),
                                    backend_factory=ProductionBackend)
        except DirectContractError as error:
            journal.finish(error.partial_record, partial=True)
            raise
        # При другом исключении начало остаётся незавершённым до явного разбора.
        return journal.finish(result)


def _require_direct(admission):
    with DirectJournal(admission.to_dict()['output'], admission, read_only=True) as journal:
        if journal.pending_exists:
            raise CheckpointError('Resolve the prepared direct record before inversion')
        record = journal.record
        if record is None or record['status'] != 'completed':
            raise CheckpointError('A completed direct-series record is required before inversion')
        result = record['result']
        admitted = admission.to_dict()
        direct = admitted.get('direct') or default_direct_spec(admission.spec)
        expected = len(direct['source_ids']) * len(direct['domains'])
        if (result.get('expected_fields') != expected
                or set(result.get('fields', {})) != set(direct['source_ids'])):
            raise CheckpointError('Direct screen does not cover the admitted fields')
        for rows in result['fields'].values():
            if set(rows) != set(direct['domains']) or any(row.get('status') not in
                    {'complete','residual_rejected','unavailable'} for row in rows.values()):
                raise CheckpointError('Direct screen contains missing or nonterminal fields')
        # Невыполнение численных порогов и порогов области записывается как результат; заданное
        # сравнение на фиксированной области не меняется.
        return record


def _group(admission, source, replicate, *, recover_pending=False):
    paths = admission.paths_for(source, replicate)
    _require_freeze(admission)
    direct_record = _require_direct(admission)
    record = admission.to_dict()
    checkpoint_path = _plain_path(Path(record['output'])/source/f'replicate_{replicate}.json',
                                  label='E06 checkpoint')
    bindings = dict(source=source, replicate=replicate,
        admission_sha256=admission.sha256,
        source_record_sha256=record['sources'][source]['sha256'],
        direct_record_sha256=digest(direct_record))
    with Checkpoint(checkpoint_path, bindings, [p.id for p in paths],
                    recover_pending=recover_pending) as checkpoint:
        if checkpoint.record['stage'] == 'scored':
            checkpoint.require_sealed()
            return checkpoint.record
        baseline = None
        if any(p.reuse is not None for p in paths):
            baseline_path = Path(record['baseline']['run_manifest_path']).parent/source/f'replicate_{replicate}.json'
            baseline = VerifiedBaseline.from_terminal(baseline_path,
                expected_bindings=admission.expected_baseline_bindings(source, replicate),
                expected_paths=admission.expected_baseline_paths(source, replicate),
                expected_file_sha256=record['baseline']['group_files'][str(baseline_path)])
        sources = _sources(admission)
        backend = ProductionBackend(admission.spec, sources[source], source_id=source)
        panels = PanelFactory(admission.spec)
        return run_group(admission.spec, paths, checkpoint, backend, panels, baseline)


def main(argv=None):
    """Выполнить выбранную команду E06 и напечатать её статус.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы без имени программы; None использует ``sys.argv[1:]``.

    Returns
    -------
    int
        Ноль после успешного выполнения команды.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help="JSON-конфигурация E06 с завершённым E05 и каталогом вывода")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('admit', help='Проверить конфигурацию, входные данные и завершённые группы E05 без записи и решения прямой или обратной задачи')
    freeze = sub.add_parser('freeze', help='Сохранить неизменяемую конфигурацию принятого расчёта')
    freeze.add_argument('--commit-pending', action='store_true', help='Завершить запись уже подготовленной конфигурации')
    direct = sub.add_parser('direct', help='Выполнить заданную серию прямых расчётов')
    direct.add_argument('--commit-pending', action='store_true', help='Завершить запись подготовленного результата без повторного расчёта')
    group = sub.add_parser('group', help='Выполнить или продолжить одну объявленную группу восстановления')
    group.add_argument('--source', required=True, choices=('PG10','EC04'), help="Источник объявленной группы восстановления")
    group.add_argument('--replicate', required=True, type=int, choices=range(1,5), help="Номер реализации шума")
    group.add_argument('--recover-pending', action='store_true', help="Завершить ранее подготовленную запись группы после проверки")
    args = parser.parse_args(argv)
    admission = build_admission(args.config)
    if args.command == 'admit':
        result = admission.to_dict()
    elif args.command == 'freeze':
        with Freeze(admission.to_dict()['output'], admission) as journal:
            result = journal.commit_pending() if args.commit_pending else journal.freeze()
    elif args.command == 'direct':
        result = _direct(admission, commit_pending=args.commit_pending)
    else:
        result = _group(admission, args.source, args.replicate,
                        recover_pending=args.recover_pending)
    print(json.dumps(dict(command=args.command, admission_sha256=admission.sha256,
        status=('frozen' if args.command=='freeze' else result.get('status', result.get('stage'))),
        output=admission.to_dict()['output']),
        ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
