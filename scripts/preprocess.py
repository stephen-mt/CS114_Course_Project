"""Build the canonical processed train and holdout files."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from student_depression.preprocessing import build_processed_splits


def main() -> None:
    train, test = build_processed_splits(
        PROJECT_ROOT / "data/raw/student_depression.csv",
        PROJECT_ROOT / "data/processed",
    )
    print(f"Wrote train.csv: {train.shape}")
    print(f"Wrote test.csv:  {test.shape}")


if __name__ == "__main__":
    main()
