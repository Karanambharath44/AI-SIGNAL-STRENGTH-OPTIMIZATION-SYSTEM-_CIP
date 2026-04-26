"""
main.py
-------
Master pipeline runner for the AI Signal Strength Optimization System.
Executes every stage in order and verifies outputs before starting Flask.
"""

import os
import sys

print("=" * 60)
print("  🛜   AI SIGNAL STRENGTH OPTIMIZATION SYSTEM")
print("=" * 60)

# Step 1 – Generate data
print("\n[1/5]  Generating signal data…")
from data_generator import generate_signal_data
df, access_points, signal_map = generate_signal_data()
os.makedirs("data", exist_ok=True)
df.to_csv("data/signal_data.csv", index=False)
print(f"       ✅  {len(df)} measurement points generated.")

# Step 2 – Analyse
print("\n[2/5]  Analysing and pre-processing…")
from signal_analyzer import analyze_signal, preprocess
analyze_signal(df)
df, label_encoder, scaler = preprocess(df)
print("       ✅  Pre-processing complete.")

# Step 3 – Train ML model
print("\n[3/5]  Training ML models…")
from ml_model import train_model
model = train_model(df)
print("       ✅  Model trained and saved.")

# Step 4 – Optimise AP placement
print("\n[4/5]  Optimising access point placement…")
from optimizer import find_optimal_placement
suggestions = find_optimal_placement()
print("       ✅  Optimisation complete.")

# Step 5 – Visualise
print("\n[5/5]  Generating visualisations…")
from visualizer import (
    plot_signal_heatmap,
    plot_quality_distribution,
    plot_predicted_vs_actual,
)
plot_signal_heatmap(signal_map, access_points)
plot_quality_distribution(df)
plot_predicted_vs_actual(model, df)
print("       ✅  All charts saved to /outputs/")

print("\n" + "=" * 60)
print("  🎉  Pipeline complete!  Starting Flask server…")
print("  🌐  Dashboard → http://localhost:5000")
print("=" * 60 + "\n")

# Start Flask
from app import app
app.run(host="0.0.0.0", port=5000, debug=False)
