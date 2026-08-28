"""Reproducible preprocessing for the student depression dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

TARGET = "Depression"
DROP_COLUMNS = ["id", "City", "Profession"]
SLEEP_COLUMN = "Sleep Duration"
SLEEP_MAP = {
    "Less than 5 hours": 4.0,
    "5-6 hours": 5.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 9.0,
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate the columns needed by the pipeline."""
    frame = pd.read_csv(path)
    required = {TARGET, SLEEP_COLUMN, *DROP_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def _prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = frame.dropna(subset=[TARGET]).drop_duplicates().copy()
    target = cleaned.pop(TARGET).astype(int)
    features = cleaned.drop(columns=DROP_COLUMNS)
    features[SLEEP_COLUMN] = features[SLEEP_COLUMN].map(SLEEP_MAP)
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a train-fitted transformer for numeric and categorical columns."""
    categorical_columns = features.select_dtypes(include=["object", "category"]).columns
    numeric_columns = features.columns.difference(categorical_columns, sort=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns.tolist()),
            ("categorical", categorical_pipeline, categorical_columns.tolist()),
        ],
        verbose_feature_names_out=False,
    )


def build_processed_splits(
    raw_path: str | Path,
    output_dir: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified, leakage-safe train and holdout CSV files."""
    features, target = _prepare_features(load_dataset(raw_path))
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    preprocessor = build_preprocessor(X_train)
    train_values = preprocessor.fit_transform(X_train)
    test_values = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    train_frame = pd.DataFrame(train_values, columns=feature_names, index=X_train.index)
    test_frame = pd.DataFrame(test_values, columns=feature_names, index=X_test.index)
    train_frame[TARGET] = y_train
    test_frame[TARGET] = y_test

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(destination / "train.csv", index=False)
    test_frame.to_csv(destination / "test.csv", index=False)
    return train_frame, test_frame
