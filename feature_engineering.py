"""Feature preparation shared by model training and API inference."""
import pandas as pd

DROP_COLS = [
    "record_id", "project_segment_id", "project_segment_name", "person_id",
    "person_name", "contact_number", "device_imei", "sim_iccid", "recorded_at",
    "alert_details_raw", "alert_message", "alert_flag", "help_status", "target",
]

def prepare_features(frame, feature_columns=None, numeric_columns=None, categorical_columns=None):
    """Prepare features, preserve schema, and convert form/JSON values to expected types."""
    data = frame.copy()

    if "recorded_at" in data.columns:
        ts = pd.to_datetime(data["recorded_at"], errors="coerce")
        data["record_hour"] = ts.dt.hour
        data["record_dayofweek"] = ts.dt.dayofweek
        data["record_month"] = ts.dt.month

    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")
    data = data.dropna(axis=1, how="all")

    if feature_columns is not None:
        for col in feature_columns:
            if col not in data.columns:
                data[col] = None
        data = data[list(feature_columns)]

    if numeric_columns is not None:
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

    if categorical_columns is not None:
        for col in categorical_columns:
            if col in data.columns:
                # Keep missing values as missing so the pipeline's imputer handles them.
                data[col] = data[col].where(data[col].notna(), None)
                data[col] = data[col].map(lambda value: str(value) if value is not None else None)

    return data
