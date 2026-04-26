"""
tests/test_pipeline.py
----------------------
Unit tests for the AI Signal Strength Optimization pipeline.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from data_generator import generate_signal_data, classify_signal
from signal_analyzer import analyze_signal, preprocess
from optimizer import find_optimal_placement


# ── Data Generator ────────────────────────────────────────────────
class TestDataGenerator:
    def test_shape(self):
        df, aps, grid = generate_signal_data(grid_size=10, num_access_points=2)
        assert len(df) == 100
        assert grid.shape == (10, 10)

    def test_rssi_range(self):
        _, _, grid = generate_signal_data()
        assert grid.min() >= -100
        assert grid.max() <= -20

    def test_access_points_count(self):
        _, aps, _ = generate_signal_data(num_access_points=4)
        assert len(aps) == 4

    def test_quality_labels(self):
        df, _, _ = generate_signal_data()
        valid = {"Excellent", "Good", "Fair", "Poor", "Very Poor"}
        assert set(df["quality"].unique()).issubset(valid)

    @pytest.mark.parametrize("rssi,expected", [
        (-45, "Excellent"),
        (-55, "Good"),
        (-65, "Fair"),
        (-75, "Poor"),
        (-85, "Very Poor"),
    ])
    def test_classify_signal(self, rssi, expected):
        assert classify_signal(rssi) == expected


# ── Analyzer ─────────────────────────────────────────────────────
class TestAnalyzer:
    def test_preprocess_columns(self):
        df, _, _ = generate_signal_data()
        df_p, le, scaler = preprocess(df)
        assert "quality_encoded" in df_p.columns
        assert "x_scaled" in df_p.columns
        assert "y_scaled" in df_p.columns

    def test_scaled_mean_approx_zero(self):
        df, _, _ = generate_signal_data()
        df_p, _, _ = preprocess(df)
        assert abs(df_p["x_scaled"].mean()) < 0.01
        assert abs(df_p["y_scaled"].mean()) < 0.01


# ── Optimizer ────────────────────────────────────────────────────
class TestOptimizer:
    def test_returns_n_suggestions(self):
        results = find_optimal_placement(n=2, grid_size=10, maxiter=10)
        assert len(results) == 2

    def test_bounds_respected(self):
        results = find_optimal_placement(n=3, grid_size=20, maxiter=20)
        for r in results:
            assert 0 <= r["x"] <= 19
            assert 0 <= r["y"] <= 19

    def test_avg_rssi_reasonable(self):
        results = find_optimal_placement(n=1, grid_size=10, maxiter=10)
        assert -100 <= results[0]["avg_rssi_dBm"] <= -20
