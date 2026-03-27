"""
Performance evaluation and visualization for all three modeling experts.

Produces:
  Regression  – RMSE, MAE, R², SNR; predicted vs actual scatter;
                residuals histogram; feature importance bar chart
  Classification – accuracy, precision, recall, F1; confusion matrix
                   heatmap; per-class accuracy summary bar chart;
                   feature importance bar chart
  Clustering  – cluster scatter plots; cluster profile heatmap
  Overall     – analysis-table correlation matrix heatmap;
                avg_fare distribution histogram
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

EVAL_PLOT_DIR = Path("eda/plots/evaluation")


def _ensure_plot_dir(subdir=None):
    """Create and return the target plot directory."""
    d = EVAL_PLOT_DIR / subdir if subdir else EVAL_PLOT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── Regression Evaluation ───────────────────────────────────────────────────

def print_regression_metrics(metrics):
    """Pretty-print regression evaluation metrics."""
    print("\n--- Regression Metrics ---")
    print(f"  RMSE : ${metrics['rmse']:.2f}")
    print(f"  MAE  : ${metrics['mae']:.2f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"  SNR  : {metrics['snr_db']:.2f} dB")


def plot_predicted_vs_actual(y_test, y_pred, title="Predicted vs Actual Fare"):
    """Scatter plot of predicted versus actual values with ideal fit line."""
    d = _ensure_plot_dir("regression")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        y_test, y_pred,
        alpha=0.4, s=18, color="steelblue", edgecolors="navy", linewidths=0.3,
    )
    lims = [
        min(float(np.min(y_test)), float(np.min(y_pred))),
        max(float(np.max(y_test)), float(np.max(y_pred))),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Fare ($)")
    ax.set_ylabel("Predicted Fare ($)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    path = d / "predicted_vs_actual.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residuals_histogram(y_test, y_pred):
    """Histogram of prediction residuals (actual − predicted)."""
    d = _ensure_plot_dir("regression")
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.set_xlabel("Residual  (Actual − Predicted, $)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals Distribution – Fare Prediction")
    ax.legend()
    plt.tight_layout()
    path = d / "residuals_histogram.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_regression_feature_importance(model, feature_names):
    """Horizontal bar chart of GBR feature importances."""
    d = _ensure_plot_dir("regression")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot.barh(ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Feature Importance – Gradient Boosting Regressor")
    ax.set_xlabel("Relative Importance")
    plt.tight_layout()
    path = d / "regression_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Classification Evaluation ───────────────────────────────────────────────

def print_classification_metrics(metrics):
    """Pretty-print classification evaluation metrics."""
    print("\n--- Classification Metrics ---")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    report = metrics["report"]
    for label, name in [("0", "Not Saturated"), ("1", "Saturated")]:
        if label in report:
            r = report[label]
            print(
                f"  {name:15s}: "
                f"Precision={r['precision']:.3f}  "
                f"Recall={r['recall']:.3f}  "
                f"F1={r['f1-score']:.3f}"
            )


def plot_confusion_matrix(
    cm,
    class_labels=("Not Saturated", "Saturated"),
    title="Confusion Matrix – Route Saturation",
):
    """Heatmap of the classification confusion matrix."""
    d = _ensure_plot_dir("classification")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = d / "confusion_matrix_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_classification_feature_importance(model, feature_names):
    """Horizontal bar chart of Random Forest feature importances."""
    d = _ensure_plot_dir("classification")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot.barh(ax=ax, color="forestgreen", edgecolor="black")
    ax.set_title("Feature Importance – Random Forest Classifier")
    ax.set_xlabel("Relative Importance")
    plt.tight_layout()
    path = d / "classification_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy_summary(metrics):
    """Grouped bar chart of precision, recall, F1 per class."""
    d = _ensure_plot_dir("classification")
    report = metrics["report"]
    rows = []
    for label, name in [("0", "Not Saturated"), ("1", "Saturated")]:
        if label in report:
            rows.append(
                {
                    "Class": name,
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1-score": report[label]["f1-score"],
                }
            )
    df_plot = pd.DataFrame(rows).set_index("Class")
    ax = df_plot.plot.bar(figsize=(8, 5), edgecolor="black", colormap="Set2", rot=0)
    ax.set_title(f"Classification Metrics per Class  (Overall Accuracy={metrics['accuracy']:.3f})")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = d / "classification_accuracy_summary.png"
    ax.figure.savefig(path)
    plt.close(ax.figure)
    print(f"  Saved: {path}")


# ─── Clustering Evaluation ───────────────────────────────────────────────────

def plot_cluster_scatter(df_clustered, x="market_distance", y="avg_fare", n_clusters=4):
    """Scatter plot coloured by K-Means cluster assignment."""
    d = _ensure_plot_dir("clustering")
    unique_clusters = sorted(df_clustered["cluster"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, c in enumerate(unique_clusters):
        sub = df_clustered[df_clustered["cluster"] == c]
        ax.scatter(
            sub[x], sub[y],
            label=f"Cluster {c}",
            alpha=0.5, s=20,
            color=palette[idx], edgecolors="none",
        )
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(
        f"K-Means Clusters: {y.replace('_', ' ').title()} "
        f"vs {x.replace('_', ' ').title()}"
    )
    ax.legend()
    plt.tight_layout()
    path = d / f"cluster_scatter_{x}_vs_{y}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cluster_profile(df_clustered, feature_cols):
    """Heatmap of normalised cluster-mean feature values (cluster profile)."""
    d = _ensure_plot_dir("clustering")
    profile = df_clustered.groupby("cluster")[feature_cols].mean()
    col_range = profile.max() - profile.min()
    profile_norm = (profile - profile.min()) / col_range.where(col_range != 0, 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        profile_norm.T,
        annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_title("Cluster Profiles – Normalised Feature Means")
    ax.set_xlabel("Cluster")
    plt.tight_layout()
    path = d / "cluster_profile_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Analysis Table Visualizations ───────────────────────────────────────────

def plot_analysis_table_correlation(df):
    """Lower-triangle correlation matrix heatmap for all numeric features."""
    d = _ensure_plot_dir()
    num_cols = [
        "avg_fare", "passengers_db1b", "market_distance",
        "passengers_t100", "seats", "departures_performed",
        "avg_fuel_price", "load_factor",
    ]
    corr = df[num_cols].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask,
        annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 8}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Analysis Table)")
    plt.tight_layout()
    path = d / "analysis_table_correlation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fare_histogram(df):
    """Histogram of average route fares with mean/median reference lines."""
    d = _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["avg_fare"].dropna(), bins=60, edgecolor="black", alpha=0.7, color="steelblue")
    mean_fare = df["avg_fare"].mean()
    median_fare = df["avg_fare"].median()
    ax.axvline(mean_fare, color="red", linestyle="--", label=f"Mean=${mean_fare:.0f}")
    ax.axvline(median_fare, color="orange", linestyle=":", label=f"Median=${median_fare:.0f}")
    ax.set_title("Distribution of Average Route Fares (Analysis Table)")
    ax.set_xlabel("Average Fare ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    path = d / "avg_fare_histogram.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Feature Selection Visualization ─────────────────────────────────────────

def plot_feature_selection_scores(reg_scores, clf_scores):
    """Side-by-side bar charts of SelectKBest F-scores for regression and classification."""
    d = _ensure_plot_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    reg_scores.sort_values().plot.barh(
        ax=axes[0], color="steelblue", edgecolor="black"
    )
    axes[0].set_title("SelectKBest F-scores\n(Regression – avg_fare)")
    axes[0].set_xlabel("F-score")

    clf_scores.sort_values().plot.barh(
        ax=axes[1], color="forestgreen", edgecolor="black"
    )
    axes[1].set_title("SelectKBest F-scores\n(Classification – is_saturated)")
    axes[1].set_xlabel("F-score")

    plt.tight_layout()
    path = d / "feature_selection_scores.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pca_explained_variance(explained):
    """Bar + cumulative line chart of PCA explained variance ratio."""
    d = _ensure_plot_dir()
    cumulative = explained.cumsum()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(explained.index, explained.values, color="steelblue", edgecolor="black", label="Explained")
    ax.plot(explained.index, cumulative.values, "ro-", label="Cumulative")
    ax.set_title("PCA Explained Variance Ratio")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    path = d / "pca_explained_variance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Master evaluation runner ─────────────────────────────────────────────────

def run_all_evaluation(results, df, reg_scores=None, clf_scores=None, pca_explained=None):
    """Run all evaluation and plotting for the three modeling experts.

    Parameters
    ----------
    results       : dict returned by modeling.run_all_models()
    df            : analysis table DataFrame
    reg_scores    : optional pd.Series of regression F-scores from feature selection
    clf_scores    : optional pd.Series of classification F-scores from feature selection
    pca_explained : optional pd.Series of PCA explained variance ratios
    """
    from modeling import REGRESSION_FEATURES, CLASSIFICATION_FEATURES, CLUSTER_FEATURES

    reg_model, X_test_r, y_test_r, y_pred_r, reg_metrics = results["regression"]
    clf_model, X_test_c, y_test_c, y_pred_c, clf_metrics = results["classification"]
    km_model, km_scaler, labels, df_clustered = results["clustering"]

    print("\n=== Regression Evaluation ===")
    print_regression_metrics(reg_metrics)
    plot_predicted_vs_actual(y_test_r, y_pred_r)
    plot_residuals_histogram(y_test_r, y_pred_r)
    plot_regression_feature_importance(reg_model, REGRESSION_FEATURES)

    print("\n=== Classification Evaluation ===")
    print_classification_metrics(clf_metrics)
    plot_confusion_matrix(clf_metrics["confusion_matrix"])
    plot_classification_feature_importance(clf_model, CLASSIFICATION_FEATURES)
    plot_accuracy_summary(clf_metrics)

    print("\n=== Clustering Evaluation ===")
    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"  Cluster sizes:\n{sizes.to_string()}")
    plot_cluster_scatter(df_clustered, "market_distance", "avg_fare", n_clusters=4)
    plot_cluster_scatter(df_clustered, "load_factor", "avg_fare", n_clusters=4)
    plot_cluster_profile(df_clustered, CLUSTER_FEATURES)

    print("\n=== Analysis Table Visualizations ===")
    plot_analysis_table_correlation(df)
    plot_fare_histogram(df)

    if reg_scores is not None and clf_scores is not None:
        print("\n=== Feature Selection Scores ===")
        plot_feature_selection_scores(reg_scores, clf_scores)

    if pca_explained is not None:
        print("\n=== PCA Explained Variance ===")
        plot_pca_explained_variance(pca_explained)


if __name__ == "__main__":
    from modeling import run_all_models, load_analysis_table
    from features import select_features_regression, select_features_classification, apply_pca, ALL_NUMERIC_FEATURES

    df = load_analysis_table()
    results = run_all_models()

    clean_df = df.dropna(subset=ALL_NUMERIC_FEATURES)
    _, reg_scores = select_features_regression(clean_df, k=5)
    _, clf_scores = select_features_classification(clean_df, k=5)
    _, _, pca_explained = apply_pca(clean_df, n_components=3)

    run_all_evaluation(results, df, reg_scores=reg_scores, clf_scores=clf_scores, pca_explained=pca_explained)
