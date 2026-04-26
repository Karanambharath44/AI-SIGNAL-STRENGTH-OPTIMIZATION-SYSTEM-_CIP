"""
data_generator.py
-----------------
Simulates RSSI (Received Signal Strength Indicator) signal data
across a 2D grid using a free-space path loss model.
"""

import numpy as np
import pandas as pd
import os


def classify_signal(rssi: float) -> str:
    """Classify an RSSI value into a quality label."""
    if rssi >= -50:
        return "Excellent"
    elif rssi >= -60:
        return "Good"
    elif rssi >= -70:
        return "Fair"
    elif rssi >= -80:
        return "Poor"
    return "Very Poor"


def generate_signal_data(
    grid_size: int = 20,
    num_access_points: int = 3,
    noise_level: float = 5.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray]:
    """
    Generate simulated RSSI signal data over a 2D grid.

    Parameters
    ----------
    grid_size        : Side length of the square grid (default 20).
    num_access_points: Number of access points to simulate (default 3).
    noise_level      : Std dev of Gaussian noise added to RSSI (default 5 dBm).
    seed             : Random seed for reproducibility (default 42).

    Returns
    -------
    df          : DataFrame with columns x, y, rssi, quality.
    access_points: List of (row, col) AP positions.
    signal_map  : 2D NumPy array of RSSI values.
    """
    rng = np.random.default_rng(seed)

    # Random AP positions
    access_points = [
        (rng.integers(0, grid_size), rng.integers(0, grid_size))
        for _ in range(num_access_points)
    ]

    signal_map = np.full((grid_size, grid_size), -100.0)

    for ax, ay in access_points:
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((i - ax) ** 2 + (j - ay) ** 2) + 0.1
                rssi = -30.0 - 20.0 * np.log10(dist)
                signal_map[i, j] = max(signal_map[i, j], rssi)

    # Add Gaussian noise
    signal_map += rng.normal(0, noise_level, signal_map.shape)
    signal_map = np.clip(signal_map, -100, -20)

    # Build DataFrame
    records = [
        {"x": i, "y": j, "rssi": round(float(signal_map[i, j]), 2),
         "quality": classify_signal(signal_map[i, j])}
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    df = pd.DataFrame(records)

    return df, access_points, signal_map


if __name__ == "__main__":
    df, aps, grid = generate_signal_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/signal_data.csv", index=False)
    print(f"✅  Signal data generated — {len(df)} points")
    print(f"📡  Access Points : {aps}")
    print(f"\n{df['quality'].value_counts().to_string()}")
