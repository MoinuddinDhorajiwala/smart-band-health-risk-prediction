"""Train and serialize the Random Forest model using the supplied CSV."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_engineering import prepare_features

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "smart_band_indian_dataset.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "training_metrics.json"
RANDOM_STATE = 42

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Put smart_band_indian_dataset.csv in the project root."
        )

    df = pd.read_csv(DATA_PATH)
    if "alert_flag" not in df.columns:
        raise ValueError("CSV must contain the alert_flag column with No/Yes labels.")

    data = df.copy()
    y = data["alert_flag"].map({"No": 0, "Yes": 1})
    valid = y.notna()
    y = y.loc[valid].astype(int)
    X = prepare_features(data.loc[valid].drop(columns=["alert_flag"], errors="ignore"))
    X = X.dropna(axis=1, how="all")
    feature_columns = X.columns.tolist()

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "model": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_columns,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "warning": "Research/demo output only; not a medically validated diagnosis."
    }

    joblib.dump({
        "model": pipeline,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
