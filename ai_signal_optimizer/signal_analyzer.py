"""
signal_analyzer.py
------------------
Statistical analysis and preprocessing of RSSI signal data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def analyze_signal(df: pd.DataFrame) -> None:
    """Print a formatted statistical analysis report."""
    print("=" * 55)
    print("  📊  SIGNAL STRENGTH ANALYSIS REPORT")
    print("=" * 55)
    print(f"  Total measurement points : {len(df)}")
    print(f"  Average RSSI             : {df['rssi'].mean():.2f} dBm")
    print(f"  Best RSSI                : {df['rssi'].max():.2f} dBm")
    print(f"  Worst RSSI               : {df['rssi'].min():.2f} dBm")
    print(f"  Std Deviation            : {df['rssi'].std():.2f} dBm")
    print(f"  Weak zones (< -75 dBm)  : {(df['rssi'] < -75).sum()}")
    print("\n  🏆  Quality Distribution:")
    print(df["quality"].value_counts().to_string())
    print("=" * 55)


def preprocess(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, StandardScaler]:
    """
    Encode quality labels and scale spatial features.

    Returns the augmented DataFrame, fitted LabelEncoder,
    and fitted StandardScaler.
    """
    le = LabelEncoder()
    df = df.copy()
    df["quality_encoded"] = le.fit_transform(df["quality"])

    scaler = StandardScaler()
    df[["x_scaled", "y_scaled"]] = scaler.fit_transform(df[["x", "y"]])

    return df, le, scaler


if __name__ == "__main__":
    from data_generator import generate_signal_data

    df, _, _ = generate_signal_data()
    analyze_signal(df)
    df, le, scaler = preprocess(df)
    print("\n✅  Preprocessing complete")
    print(df[["x", "y", "rssi", "quality", "quality_encoded"]].head(8))
