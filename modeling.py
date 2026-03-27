"""
Three modeling experts for airfare analysis.

Expert 1 – Gradient Boosting Regressor  : predict avg_fare  (regression)
Expert 2 – Random Forest Classifier     : predict is_saturated (classification)
Expert 3 – K-Means Clustering           : unsupervised route grouping
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf

CLEANED_DIR = Path("cleaned_data")

# Features used for regression (predicting avg_fare)
REGRESSION_FEATURES = [
    "market_distance",
    "load_factor",
    "avg_fuel_price",
    "passengers_db1b",
    "departures_performed",
    "seats",
]

# Features used for classification (predicting is_saturated)
CLASSIFICATION_FEATURES = [
    "avg_fare",
    "market_distance",
    "passengers_db1b",
    "departures_performed",
    "avg_fuel_price",
]

# Features used for clustering
CLUSTER_FEATURES = [
    "avg_fare",
    "market_distance",
    "load_factor",
    "passengers_db1b",
    "departures_performed",
]


def load_analysis_table():
    """Load the analysis table and drop rows missing load_factor."""
    df = pd.read_csv(CLEANED_DIR / "analysis_table.csv")
    df = df.dropna(subset=["load_factor"])
    return df


# ─── Expert 1: Gradient Boosting Regressor ───────────────────────────────────

def _regression_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R², and SNR for regression results."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    signal_power = float(np.mean(np.array(y_true) ** 2))
    noise_power = float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    snr_db = 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    return {"rmse": rmse, "mae": mae, "r2": r2, "snr_db": float(snr_db)}


def train_regression_model(df):
    """Train a Gradient Boosting Regressor to predict average airfare.

    Returns
    -------
    model, X_train, X_test, y_train, y_test, y_pred, metrics
    """
    X = df[REGRESSION_FEATURES].copy()
    y = df["avg_fare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _regression_metrics(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, metrics


# ─── Expert 2: Random Forest Classifier ──────────────────────────────────────

def _classification_metrics(y_true, y_pred):
    """Compute accuracy, per-class metrics, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def train_classification_model(df):
    """Train a Random Forest Classifier to predict route saturation.

    Returns
    -------
    model, X_train, X_test, y_train, y_test, y_pred, metrics
    """
    X = df[CLASSIFICATION_FEATURES].copy()
    y = df["is_saturated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _classification_metrics(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, metrics


# ─── Expert 3: K-Means Clustering (unsupervised) ─────────────────────────────

def train_clustering_model(df, n_clusters=4):
    """Cluster routes using K-Means on scaled route features.

    Returns
    -------
    model, scaler, X_scaled, labels, df_with_cluster_column
    """
    X = df[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["cluster"] = labels

    return model, scaler, X_scaled, labels, df_out


# ─── Convenience wrapper for dashboard ───────────────────────────────────────

def predict_fares(input_df):
    """Load the full dataset, train the GBR model, predict on input_df."""
    df_full = load_analysis_table()
    model, *_ = train_regression_model(df_full)
    return model.predict(input_df[REGRESSION_FEATURES])


# ─── Run all three experts ────────────────────────────────────────────────────

def run_all_models():
    """Train all three modeling experts and return their outputs."""
    df = load_analysis_table()

    print("\n=== Expert 1: Gradient Boosting Regressor (Fare Prediction) ===")
    reg_results = train_regression_model(df)
    reg_model, _, X_test_r, _, y_test_r, y_pred_r, reg_metrics = reg_results
    print(f"  RMSE  : ${reg_metrics['rmse']:.2f}")
    print(f"  MAE   : ${reg_metrics['mae']:.2f}")
    print(f"  R²    : {reg_metrics['r2']:.4f}")
    print(f"  SNR   : {reg_metrics['snr_db']:.2f} dB")

    print("\n=== Expert 2: Random Forest Classifier (Route Saturation) ===")
    clf_results = train_classification_model(df)
    clf_model, _, X_test_c, _, y_test_c, y_pred_c, clf_metrics = clf_results
    print(f"  Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"  Confusion Matrix:\n{clf_metrics['confusion_matrix']}")

    print("\n=== Expert 3: K-Means Clustering (Route Groups) ===")
    km_model, km_scaler, X_scaled, labels, df_clustered = train_clustering_model(df)
    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"  Cluster sizes: {sizes.to_dict()}")

    return {
        "regression": (reg_model, X_test_r, y_test_r, y_pred_r, reg_metrics),
        "classification": (clf_model, X_test_c, y_test_c, y_pred_c, clf_metrics),
        "clustering": (km_model, km_scaler, labels, df_clustered),
    }


if __name__ == "__main__":
    run_all_models()
