"""
optimizer.py
------------
Finds optimal Wi-Fi Access Point placements using
SciPy's Differential Evolution algorithm.
"""

import numpy as np
from scipy.optimize import differential_evolution


def _coverage_score(pos: np.ndarray, grid_size: int = 20) -> float:
    """
    Objective function: returns the *negative* mean RSSI
    (we minimise, so minimising a negative = maximising RSSI).
    """
    ax, ay = pos
    total = sum(
        -30.0 - 20.0 * np.log10(np.sqrt((i - ax) ** 2 + (j - ay) ** 2) + 0.1)
        for i in range(grid_size)
        for j in range(grid_size)
    )
    return -total / (grid_size ** 2)


def find_optimal_placement(
    n: int = 3,
    grid_size: int = 20,
    maxiter: int = 100,
    tol: float = 0.01,
) -> list[dict]:
    """
    Run Differential Evolution n times with different seeds and
    return the top-n distinct AP placement recommendations.

    Returns
    -------
    List of dicts: {suggestion, x, y, avg_rssi_dBm, improvement_dBm}
    """
    print("🔧  Running Signal Optimization Engine…")
    bounds = [(0, grid_size - 1), (0, grid_size - 1)]
    results = []

    for i in range(n):
        res = differential_evolution(
            _coverage_score,
            bounds,
            args=(grid_size,),
            seed=i * 10,
            maxiter=maxiter,
            tol=tol,
        )
        results.append(
            {
                "suggestion": i + 1,
                "x": round(float(res.x[0]), 1),
                "y": round(float(res.x[1]), 1),
                "avg_rssi_dBm": round(-res.fun, 2),
            }
        )

    print("\n✅  OPTIMAL ACCESS POINT PLACEMENT SUGGESTIONS")
    print("-" * 52)
    for r in results:
        print(
            f"  📡  Option {r['suggestion']} → Place AP at "
            f"({r['x']}, {r['y']})  |  Avg RSSI: {r['avg_rssi_dBm']} dBm"
        )
    return results


if __name__ == "__main__":
    find_optimal_placement()
