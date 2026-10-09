"""Сравнение форм источника или построение его сохранённых графиков."""
import argparse
from pathlib import Path
import json
from .run import main as run_main
from .figures import render

def main():
    """Обработать команду запуска опыта или построения сохранённых рисунков."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Выполнить или продолжить расчёт сравнения регуляризаций")
    run.add_argument("--config", required=True, help="JSON-конфигурация выбранной команды")
    run.add_argument("--source", help="Один источник из конфигурации; без аргумента выполняется весь список")
    plots = sub.add_parser("figures", help="Построить рисунки по сохранённым решениям")
    plots.add_argument("--config", required=True, help="JSON-конфигурация выбранной команды")
    args = parser.parse_args()
    if args.command == "run":
        argv = ["--config", args.config]
        if args.source is not None:
            argv += ["--source", args.source]
        return run_main(argv)
    path = Path(args.config).resolve()
    spec = json.loads(path.read_bytes())
    resolve = lambda key: (path.parent / spec[key]).resolve()
    output = resolve("output")
    protected = [Path(__file__).resolve().parents[2],resolve("input"),
                 *(path.parent/entry for entry in spec.get("protected_roots", []))]
    if any(output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output)
           for root in protected):
        raise ValueError("Output must not overlap source or protected input directories")
    return render(resolve("simulation_config"), resolve("input"), resolve("output"))

if __name__ == "__main__":
    main()
