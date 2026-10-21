"""Flask web application and JSON API for the Smart Band ML Mini Project."""
from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template
from feature_engineering import prepare_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "training_metrics.json"

app = Flask(__name__)
_bundle = None

TEAM = [
    {"name": "Faisal Alvi", "roll_no": "231202"},
    {"name": "Zaid Mirkar", "roll_no": "231236"},
    {"name": "Ibrahim Maniyar", "roll_no": "231234"},
]

def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "model.joblib is missing. Run `python train_model.py` before starting the app."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

def get_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

@app.get("/")
def index():
    metrics = get_metrics()
    return render_template("index.html", metrics=metrics, team=TEAM)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})

@app.get("/metadata")
def metadata():
    metrics = get_metrics()
    if not metrics:
        return jsonify({"message": "Training metrics not found. Run python train_model.py."}), 404
    return jsonify(metrics)

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400
    record = payload.get("features", payload)
    if not isinstance(record, dict):
        return jsonify({"error": "`features` must be a JSON object."}), 400

    try:
        bundle = get_bundle()
        frame = pd.DataFrame([record])
        X = prepare_features(
            frame,
            feature_columns=bundle["feature_columns"],
            numeric_columns=bundle.get("numeric_columns"),
            categorical_columns=bundle.get("categorical_columns"),
        )
        model = bundle["model"]
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        classes = list(model.named_steps["classifier"].classes_)
        probability_by_class = {
            ("alert" if int(cls) == 1 else "no_alert"): round(float(prob), 6)
            for cls, prob in zip(classes, probabilities)
        }
        return jsonify({
            "prediction": prediction,
            "alert_flag": "Yes" if prediction == 1 else "No",
            "risk_label": "Alert predicted" if prediction == 1 else "No alert predicted",
            "class_probabilities": probability_by_class,
            "disclaimer": "Educational research output only. Not a medical device or diagnosis."
        })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Could not process this prediction. Check the submitted values."}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
