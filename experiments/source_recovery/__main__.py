"""Запуск восстановления источника по заданной конфигурации."""
from __future__ import annotations

import argparse
from .run import run_experiment


def main():
    """Прочитать параметры командной строки и запустить восстановление источника."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True, help="JSON-конфигурация восстановления источника")
    parser.add_argument("--source", help="Один источник из конфигурации; без аргумента выполняется весь список")
    parser.add_argument("--replicate",type=int, help="Номер реализации шума; без аргумента выполняются все заданные повторы")
    args = parser.parse_args()
    run_experiment(args.config,source_id=args.source,replicate=args.replicate)


if __name__ == "__main__":
    main()
