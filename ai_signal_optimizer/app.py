"""
app.py
------
Flask REST API server for the AI Signal Strength Optimization System.
Serves the frontend dashboard and exposes all ML/optimizer endpoints.

Endpoints
---------
GET  /                  – Serve the dashboard HTML.
GET  /api/signal-map    – Full 20×20 RSSI grid as JSON.
GET  /api/stats         – Signal statistics and quality distribution.
GET  /api/optimize      – 3 optimal AP placement suggestions.
POST /api/predict       – Predict RSSI at a given (x, y) position.
GET  /api/history       – Last 20 prediction log entries.
"""

import os
import csv
import datetime
from flask import Flask, jsonify, request, render_template

from data_generator import generate_signal_data, classify_signal
from signal_analyzer import analyze_signal, preprocess
from ml_model import train_model, load_model, predict_signal
from optimizer import find_optimal_placement

app = Flask(__name__)

# ── Bootstrap once on startup ────────────────────────────────────────
print("🚀  Bootstrapping AI Signal Optimizer…")
_df, _aps, _signal_map = generate_signal_data()
analyze_signal(_df)

# Train or load model
MODEL_PATH = os.path.join("models", "signal_model.pkl")
if os.path.exists(MODEL_PATH):
    _model = load_model()
    print("✅  Loaded existing model from disk.")
else:
    _model = train_model(_df)

PREDICTION_LOG = os.path.join("data", "prediction_log.csv")
os.makedirs("data", exist_ok=True)
if not os.path.exists(PREDICTION_LOG):
    with open(PREDICTION_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "x", "y", "rssi", "quality"])


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/signal-map")
def api_signal_map():
    return jsonify({"grid": _signal_map.tolist(), "access_points": _aps})


@app.route("/api/stats")
def api_stats():
    stats = {
        "total_points": len(_df),
        "avg_rssi": round(float(_df["rssi"].mean()), 2),
        "best_rssi": round(float(_df["rssi"].max()), 2),
        "worst_rssi": round(float(_df["rssi"].min()), 2),
        "std_dev": round(float(_df["rssi"].std()), 2),
        "weak_zones": int((_df["rssi"] < -75).sum()),
        "quality_distribution": _df["quality"].value_counts().to_dict(),
    }
    return jsonify(stats)


@app.route("/api/optimize")
def api_optimize():
    suggestions = find_optimal_placement()
    return jsonify({"suggestions": suggestions})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    x = max(0, min(19, int(data.get("x", 0))))
    y = max(0, min(19, int(data.get("y", 0))))
    rssi    = predict_signal(_model, x, y)
    quality = classify_signal(rssi)

    # Log prediction
    with open(PREDICTION_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.datetime.utcnow().isoformat(), x, y, rssi, quality
        ])

    return jsonify({"x": x, "y": y, "rssi": rssi, "quality": quality})


@app.route("/api/history")
def api_history():
    rows = []
    if os.path.exists(PREDICTION_LOG):
        with open(PREDICTION_LOG) as f:
            reader = csv.DictReader(f)
            rows = list(reader)[-20:]
    return jsonify({"history": rows})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
