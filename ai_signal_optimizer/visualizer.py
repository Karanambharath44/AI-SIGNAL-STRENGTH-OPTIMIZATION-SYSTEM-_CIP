"""
visualizer.py
-------------
Generate signal heatmaps, quality-distribution charts,
and prediction scatter plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_signal_heatmap(
    signal_map: np.ndarray,
    access_points: list,
    title: str = "RSSI Signal Strength Heatmap",
) -> None:
    """Render the 2-D RSSI heatmap with AP locations marked."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        signal_map, cmap="RdYlGn", vmin=-100, vmax=-20,
        cbar_kws={"label": "RSSI (dBm)"},
        linewidths=0, ax=ax,
    )
    for ax_pos, ay_pos in access_points:
        ax.plot(ay_pos + 0.5, ax_pos + 0.5, "b^", markersize=12,
                markeredgecolor="white", label="Access Point")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Y Coordinate")
    ax.set_ylabel("X Coordinate")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "signal_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅  Heatmap saved → {path}")


def plot_quality_distribution(df) -> None:
    """Bar chart of signal-quality zone counts."""
    order  = ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
    colors = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444", "#7c3aed"]
    counts = [df["quality"].value_counts().get(q, 0) for q in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, counts, color=colors, edgecolor="black")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(cnt),
                ha="center", fontweight="bold")
    ax.set_title("Signal Quality Zone Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Grid Points")
    ax.set_xlabel("Signal Quality Category")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "quality_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅  Quality chart saved → {path}")


def plot_predicted_vs_actual(model, df) -> None:
    """Scatter plot of predicted vs actual RSSI values."""
    X = df[["x", "y"]].values
    y_actual = df["rssi"].values
    y_pred   = model.predict(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_actual, y_pred, alpha=0.4, color="steelblue",
               edgecolors="k", s=20)
    lims = [-100, -20]
    ax.plot(lims, lims, "r--", label="Ideal Prediction")
    ax.set_xlabel("Actual RSSI (dBm)")
    ax.set_ylabel("Predicted RSSI (dBm)")
    ax.set_title("Predicted vs Actual RSSI (Random Forest)", fontsize=13,
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pred_vs_actual.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅  Prediction chart saved → {path}")
