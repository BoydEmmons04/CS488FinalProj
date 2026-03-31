"""
Evaluation module: Computes performance metrics, visualizations, and analysis for all models.
Implements regression evaluation metrics required by the project rubric including RMSE, R2, MAPE, and SNR.
Generates core visualizations aligned to the project workflow: model comparison plots,
key-variable histograms, predicted-vs-actual plots, PCA variance, and time-series trends.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


OUTPUT_DIR = Path("outputs") / "evaluation"

# Legacy evaluation artifacts from removed models/plots.
DEPRECATED_EVAL_GLOBS = [
	"*gradient_boosting*",
	"residuals_*.png",
	"feature_distributions.png",
	"target_distribution.png",
	"feature_importance_gb.png",
	"feature_importance_gradient_boosting.csv",
	"feature_importance_permutation.csv",
	"pca_component_loadings.csv",
	"rmse_comparison.png",
]


def _compute_metrics(y_true, y_pred):
	"""Return standard regression metrics required in the project rubric.
	
	Metrics:
	- RMSE: Root Mean Squared Error
	- MAPE: Mean Absolute Percentage Error
	- R2: Coefficient of Determination
	- SNR: Signal-to-Noise Ratio (ratio of signal power to noise power)
	"""
	rmse = mean_squared_error(y_true, y_pred) ** 0.5
	mape = mean_absolute_percentage_error(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	
	# SNR: Signal power / Noise power
	# Signal power = mean(y_true)^2
	# Noise power = mean((y_true - y_pred)^2) = MSE
	signal_power = np.mean(y_true) ** 2
	noise_power = mean_squared_error(y_true, y_pred)
	snr = signal_power / (noise_power + 1e-8)  # Avoid division by zero
	
	return {
		"RMSE": rmse,
		"MAPE": mape,
		"R2": r2,
		"SNR": snr,
	}


def _add_accuracy_columns(metrics_df):
	"""Add explicit accuracy-style fields for easier non-technical consumption."""
	# Accuracy derived from percentage error. Clipped to [0, 100] for readability.
	metrics_df["AccuracyPct"] = ((1.0 - metrics_df["MAPE"]) * 100.0).clip(lower=0.0, upper=100.0)
	# R2 also represented as a percentage for dashboard/report tables.
	metrics_df["R2Pct"] = (metrics_df["R2"] * 100.0).clip(lower=0.0, upper=100.0)
	return metrics_df


def _save_time_plots(model_df):
	"""Save fare/load-factor over available time periods (handles Q1-only data)."""
	if model_df is None or model_df.empty:
		return

	time_df = (
		model_df.groupby(["YEAR", "QUARTER"], as_index=False)
		.agg(
			avg_fare=("avg_fare", "mean"),
			avg_load_factor=("load_factor", "mean"),
			avg_fuel_price=("avg_fuel_price", "mean"),
		)
		.sort_values(["YEAR", "QUARTER"])
	)

	time_df["period"] = (
		time_df["YEAR"].astype(int).astype(str)
		+ "-Q"
		+ time_df["QUARTER"].astype(int).astype(str)
	)
	time_df.to_csv(OUTPUT_DIR / "time_series_summary.csv", index=False)

	fig, ax1 = plt.subplots(figsize=(10, 5))
	period_idx = np.arange(len(time_df))
	line_style = "-o" if len(time_df) > 1 else "o"

	ax1.plot(period_idx, time_df["avg_fare"], line_style, color="#1f77b4", label="Avg Fare")
	ax1.set_ylabel("Average Fare ($)", color="#1f77b4")
	ax1.tick_params(axis="y", labelcolor="#1f77b4")
	ax1.set_xticks(period_idx)
	ax1.set_xticklabels(time_df["period"], rotation=20)
	ax1.set_xlabel("Time Period")

	ax2 = ax1.twinx()
	ax2.plot(
		period_idx,
		time_df["avg_load_factor"],
		line_style,
		color="#d62728",
		label="Avg Load Factor",
	)
	ax2.set_ylabel("Average Load Factor", color="#d62728")
	ax2.tick_params(axis="y", labelcolor="#d62728")

	title = "Average Fare and Load Factor Over Time"
	if len(time_df) == 1:
		title += " (Q1-Only Scope)"
	ax1.set_title(title)
	fig.tight_layout()
	fig.savefig(OUTPUT_DIR / "fare_load_factor_over_time.png", dpi=150)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(period_idx, time_df["avg_fuel_price"], line_style, color="#2ca02c")
	ax.set_xticks(period_idx)
	ax.set_xticklabels(time_df["period"], rotation=20)
	ax.set_xlabel("Time Period")
	ax.set_ylabel("Average Fuel Price")
	fuel_title = "Fuel Price Over Time"
	if len(time_df) == 1:
		fuel_title += " (Q1-Only Scope)"
	ax.set_title(fuel_title)
	fig.tight_layout()
	fig.savefig(OUTPUT_DIR / "fuel_price_over_time.png", dpi=150)
	plt.close(fig)


def _save_feature_importance_artifacts(trained_models, X_test):
	"""Save concise interpretable artifacts for linear and PCA-based models."""
	feature_cols = list(X_test.columns)

	if "linear_regression" in trained_models:
		linear_model = trained_models["linear_regression"]
		if hasattr(linear_model, "coef_"):
			coef_df = pd.DataFrame(
				{
					"feature": feature_cols,
					"coefficient": linear_model.coef_,
					"abs_coefficient": np.abs(linear_model.coef_),
				}
			).sort_values("abs_coefficient", ascending=False)
			coef_df.to_csv(OUTPUT_DIR / "feature_importance_linear.csv", index=False)

	if "pca_regression" in trained_models:
		pca_pipeline = trained_models["pca_regression"]
		pca_model = pca_pipeline.named_steps.get("pca")
		if pca_model is not None and hasattr(pca_model, "components_"):
			component_cols = [f"PC{i + 1}" for i in range(pca_model.n_components_)]
			explained_df = pd.DataFrame(
				{
					"component": component_cols,
					"explained_variance_ratio": pca_model.explained_variance_ratio_,
					"cumulative_explained_variance": np.cumsum(pca_model.explained_variance_ratio_),
				}
			)
			explained_df.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)

			fig, ax = plt.subplots(figsize=(8, 5))
			ax.plot(
				range(1, len(explained_df) + 1),
				explained_df["cumulative_explained_variance"],
				marker="o",
			)
			ax.set_xlabel("Number of Components")
			ax.set_ylabel("Cumulative Explained Variance")
			ax.set_title("PCA Cumulative Explained Variance")
			ax.set_ylim([0, 1.05])
			plt.tight_layout()
			fig.savefig(OUTPUT_DIR / "pca_explained_variance.png", dpi=150)
			plt.close(fig)


def evaluate_model_outputs(model_results, save=True):
	"""Evaluate modeled predictions and save core project artifacts.
	
	Generates:
	- Performance metrics (RMSE, R2, MAPE, SNR)
	- Model comparison charts (RMSE/SNR, R2/MAPE, accuracy)
	- Predicted-vs-actual plots for both core models
	- Key-variable histograms for hypothesis variables
	- Linear coefficient and PCA explained-variance artifacts
	- Time-series trend plots
	"""
	y_test = model_results["y_test"]
	predictions = model_results["predictions"]
	trained_models = model_results["trained_models"]
	X_test = model_results["X_test"]
	model_df = model_results.get("model_df")

	metric_rows = []
	for model_name, y_pred in predictions.items():
		row = {"model": model_name}
		row.update(_compute_metrics(y_test, y_pred))
		metric_rows.append(row)

	metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
	metrics_df = _add_accuracy_columns(metrics_df)

	if save:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

		# Remove stale files so outputs clearly map to the final model set.
		for pattern in DEPRECATED_EVAL_GLOBS:
			for path in OUTPUT_DIR.glob(pattern):
				if path.exists():
					path.unlink()

		metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

		# 1. Actual vs predicted scatter for each model
		for model_name, y_pred in predictions.items():
			fig, ax = plt.subplots(figsize=(8, 6))
			ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="black", linewidths=0.2)
			min_val = min(float(y_test.min()), float(y_pred.min()))
			max_val = max(float(y_test.max()), float(y_pred.max()))
			ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
			ax.set_xlabel("Actual Avg Fare ($)")
			ax.set_ylabel("Predicted Avg Fare ($)")
			ax.set_title(f"Actual vs Predicted: {model_name}")
			plt.tight_layout()
			fig.savefig(OUTPUT_DIR / f"actual_vs_predicted_{model_name}.png", dpi=150)
			plt.close(fig)

		# 2. RMSE and SNR comparison chart
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
		
		colors = ["#4C78A8", "#F58518", "#54A24B"]
		ax1.bar(metrics_df["model"], metrics_df["RMSE"], color=colors)
		ax1.set_title("Model RMSE Comparison (Lower is Better)")
		ax1.set_ylabel("RMSE ($)")
		ax1.set_xlabel("Model")
		ax1.tick_params(axis='x', rotation=20)
		
		ax2.bar(metrics_df["model"], metrics_df["SNR"], color=colors)
		ax2.set_title("Model SNR Comparison (Higher is Better)")
		ax2.set_ylabel("Signal-to-Noise Ratio")
		ax2.set_xlabel("Model")
		ax2.tick_params(axis='x', rotation=20)
		
		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / "rmse_snr_comparison.png", dpi=150)
		plt.close(fig)

		# 3. R2 and MAPE comparison
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
		
		ax1.bar(metrics_df["model"], metrics_df["R2"], color=colors)
		ax1.set_title("Model R² Comparison (Higher is Better)")
		ax1.set_ylabel("R² Score")
		ax1.set_xlabel("Model")
		ax1.set_ylim([0, 1])
		ax1.tick_params(axis='x', rotation=20)
		
		ax2.bar(metrics_df["model"], metrics_df["MAPE"], color=colors)
		ax2.set_title("Model MAPE Comparison (Lower is Better)")
		ax2.set_ylabel("Mean Absolute Percentage Error")
		ax2.set_xlabel("Model")
		ax2.tick_params(axis='x', rotation=20)
		
		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / "r2_mape_comparison.png", dpi=150)
		plt.close(fig)

		# 4. User-facing model accuracy comparison
		fig, ax = plt.subplots(figsize=(10, 5))
		ax.bar(metrics_df["model"], metrics_df["AccuracyPct"], color="#2E8B57")
		ax.set_title("Model Accuracy % (100 - MAPE)")
		ax.set_ylabel("Accuracy (%)")
		ax.set_xlabel("Model")
		ax.set_ylim([0, 100])
		ax.tick_params(axis="x", rotation=20)
		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
		plt.close(fig)

		# 5. Key-variable histograms aligned to project hypothesis variables.
		if model_df is not None and not model_df.empty:
			hist_cols = ["avg_fare", "load_factor", "avg_fuel_price", "passengers_db1b"]
			hist_cols = [c for c in hist_cols if c in model_df.columns]
			fig, axes = plt.subplots(2, 2, figsize=(12, 8))
			axes = axes.flatten()
			for idx, col in enumerate(hist_cols):
				axes[idx].hist(model_df[col], bins=35, edgecolor="black", alpha=0.75, color="#4C78A8")
				axes[idx].set_title(f"Distribution: {col}")
				axes[idx].set_xlabel(col)
				axes[idx].set_ylabel("Frequency")
			for idx in range(len(hist_cols), 4):
				axes[idx].set_visible(False)
			plt.tight_layout()
			fig.savefig(OUTPUT_DIR / "key_variable_histograms.png", dpi=150)
			plt.close(fig)

		# 6. Linear/PCA artifact tables and plot.
		_save_feature_importance_artifacts(trained_models, X_test)

		# 7. Time-aware plots for fare/load-factor/fuel trend outputs
		_save_time_plots(model_df)

		print("All evaluation visualizations saved to:", OUTPUT_DIR)

	return metrics_df