"""Train the selected baselines and write one comparable metrics table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from student_depression.evaluation import classification_metrics, metrics_frame
from student_depression.models import ScratchLogisticRegression

TARGET = "Depression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-scratch-logistic",
        action="store_true",
        help="also evaluate the educational NumPy logistic regression",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(PROJECT_ROOT / "data/processed/train.csv")
    test = pd.read_csv(PROJECT_ROOT / "data/processed/test.csv")
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]

    models = {
        "Logistic Regression": LogisticRegression(
            C=10,
            max_iter=2_000,
            random_state=42,
        ),
        "Linear SVM": SVC(kernel="linear", C=1, random_state=42),
        "RBF SVM": SVC(kernel="rbf", C=1, gamma=0.1, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            min_samples_leaf=15,
            min_samples_split=2,
            random_state=42,
        ),
    }
    if args.include_scratch_logistic:
        models["Logistic Regression (NumPy)"] = ScratchLogisticRegression(
            learning_rate=0.1,
            max_iter=5_000,
            l2=0.001,
        )

    rows = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        rows.append(classification_metrics(name, y_test, model.predict(X_test)))

    metrics = metrics_frame(rows)
    output_path = PROJECT_ROOT / "results/model_metrics.csv"
    metrics.to_csv(output_path, index=False)
    print(f"\n{metrics.to_string(index=False)}")
    print(f"\nWrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
