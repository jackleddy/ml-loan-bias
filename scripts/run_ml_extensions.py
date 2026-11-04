"""Run the machine-learning extension studies (see loanfair/ml_extensions.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loanfair.cli import main

if __name__ == "__main__":
    raise SystemExit(main("ml_extensions"))
