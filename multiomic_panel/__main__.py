"""CLI entry: `python -m multiomic_panel --config config.yaml`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflow import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe multiomic panel discovery.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML configuration path.")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
