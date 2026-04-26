"""
ml_model.py
-----------
Train, evaluate, and persist ML models for RSSI prediction.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train_model(
    df: pd.DataFrame,
    model_dir: str = "models",
    test_size: float = 0.2,
    seed: int = 42,
) -> RandomForestRegressor:
    """
    Train Random Forest and Gradient Boosting regressors.
    Saves the best model to disk and returns it.
    """
    os.makedirs(model_dir, exist_ok=True)

    X = df[["x", "y"]].values
    y = df["rssi"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=seed),
    }

    best_model, best_r2 = None, -np.inf

    for name, model in models.items():
        print(f"🤖  Training {name}…")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"    RMSE : {rmse:.4f} dBm   R² : {r2:.4f}")
        if r2 > best_r2:
            best_r2, best_model = r2, model

    path = os.path.join(model_dir, "signal_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n✅  Best model saved → {path}  (R²={best_r2:.4f})")
    return best_model


def load_model(model_dir: str = "models") -> object:
    """Load the serialised model from disk."""
    path = os.path.join(model_dir, "signal_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_signal(model, x: int, y: int) -> float:
    """Return predicted RSSI (dBm) for grid position (x, y)."""
    return round(float(model.predict([[x, y]])[0]), 2)


if __name__ == "__main__":
    from data_generator import generate_signal_data

    df, _, _ = generate_signal_data()
    model = train_model(df)
    print(f"\n🔍  Predicted RSSI at (5,5)   : {predict_signal(model, 5, 5)} dBm")
    print(f"🔍  Predicted RSSI at (10,10) : {predict_signal(model, 10, 10)} dBm")
