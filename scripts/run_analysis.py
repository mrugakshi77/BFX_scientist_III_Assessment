"""Command-line entry point for the multiomic panel workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiomic_panel.workflow import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe multiomic panel discovery.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML configuration path.")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
