"""Сопоставление заданной модели с исходными измерениями КрАЗ."""
import argparse
from .run import run

def main():
    """Выполнить сопоставление КрАЗ по конфигурации из командной строки."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON данных, модели, периодов поиска и нового каталога результата")
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()
