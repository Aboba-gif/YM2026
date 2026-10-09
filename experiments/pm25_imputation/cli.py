"""Заполнение пропусков PM₂.₅ и экспорт по заданной конфигурации."""
import argparse
import json
from pathlib import Path

def main(argv=None):
    """Разобрать аргументы и выполнить экспорт или проверку PM₂.₅.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы без имени программы; None использует ``sys.argv[1:]``.

    Returns
    -------
    int
        0 после успешного выполнения.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help="JSON-конфигурация опыта; пути разрешаются от её каталога")
    parser.add_argument('--data-root', type=Path, help="Каталог исходных годовых CSV четырёх постов")
    parser.add_argument('--output-dir', type=Path, help="Каталог заполненного ряда, журнала пропусков и манифеста")
    parser.add_argument('--results-dir', type=Path, help="Каталог проверочных блоков, метрик и записей обучения")
    parser.add_argument('--protocol', type=Path, help="JSON-протокол выбора и оценки методов заполнения")
    parser.add_argument('--verify-only', action='store_true', help="Проверить существующий итоговый CSV без обучения и нового экспорта")
    parser.add_argument('--contract', type=Path, help="Файл пояснений, копируемый в README.md результата")
    parser.add_argument('--protected-root', type=Path,action='append',default=[], help="Дополнительный входной путь, защищённый от записи; можно повторять")
    args = parser.parse_args(argv)
    if args.config:
        if any(value is not None for value in (args.data_root,args.output_dir,args.results_dir,
                                               args.protocol,args.contract)) or args.protected_root:
            parser.error("--config нельзя сочетать с явно заданными входными и выходными путями")
        config_path = args.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        def location(key):
            return (config_path.parent/config[key]).resolve()
        args.data_root, args.output_dir, args.results_dir = (
            location("data_root"),location("output"),location("metrics"))
        args.protocol = location("protocol") if "protocol" in config else None
        args.contract = location("contract") if "contract" in config else None
        protected = [Path(__file__).resolve().parents[2],
                     *(config_path.parent/value for value in config.get("protected_roots", []))]
        args.protected_root = [root.resolve() for root in protected]
        for output in (args.output_dir,args.results_dir):
            if any(output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output)
                   for root in protected):
                raise ValueError("Output must not overlap source or protected input directories")
    elif any(value is None for value in (args.data_root,args.output_dir,args.results_dir)):
        parser.error("Передайте --config либо все три аргумента: --data-root, --output-dir и --results-dir")
    from experiments.pm25_imputation.export import run
    run(args.data_root, output_dir=args.output_dir, results_dir=args.results_dir,
        protocol_path=args.protocol, verify_only=args.verify_only, contract=args.contract,
        protected_roots=args.protected_root)
    return 0
