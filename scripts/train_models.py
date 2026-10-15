"""Train and evaluate pricing models and sensitivity variants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loanfair.cli import main

if __name__ == "__main__":
    raise SystemExit(main("train_models"))
