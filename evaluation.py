from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


OUTPUT_DIR = Path("outputs") / "evaluation"


def _compute_metrics(y_true, y_pred):
	"""Return standard regression metrics required in the project rubric."""
	return {
		"RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
		"MAPE": mean_absolute_percentage_error(y_true, y_pred),
		"R2": r2_score(y_true, y_pred),
	}


def evaluate_model_outputs(model_results, save=True):
	"""Evaluate all modeled predictions and optionally save tables/plots."""
	y_test = model_results["y_test"]
	predictions = model_results["predictions"]

	metric_rows = []
	for model_name, y_pred in predictions.items():
		row = {"model": model_name}
		row.update(_compute_metrics(y_test, y_pred))
		metric_rows.append(row)

	metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)

	if save:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

		# Actual vs predicted scatter for each model.
		for model_name, y_pred in predictions.items():
			fig, ax = plt.subplots(figsize=(8, 6))
			ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="black", linewidths=0.2)
			min_val = min(float(y_test.min()), float(y_pred.min()))
			max_val = max(float(y_test.max()), float(y_pred.max()))
			ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
			ax.set_xlabel("Actual Avg Fare")
			ax.set_ylabel("Predicted Avg Fare")
			ax.set_title(f"Actual vs Predicted: {model_name}")
			plt.tight_layout()
			fig.savefig(OUTPUT_DIR / f"actual_vs_predicted_{model_name}.png", dpi=150)
			plt.close(fig)

		# Residual plot for each model.
		for model_name, y_pred in predictions.items():
			residuals = y_test - y_pred
			fig, ax = plt.subplots(figsize=(8, 5))
			ax.hist(residuals, bins=40, edgecolor="black", alpha=0.8)
			ax.axvline(0, color="black", linestyle="--", linewidth=1)
			ax.set_xlabel("Residual (Actual - Predicted)")
			ax.set_ylabel("Frequency")
			ax.set_title(f"Residual Distribution: {model_name}")
			plt.tight_layout()
			fig.savefig(OUTPUT_DIR / f"residuals_{model_name}.png", dpi=150)
			plt.close(fig)

		# Comparison chart for the key metric.
		fig, ax = plt.subplots(figsize=(8, 5))
		ax.bar(metrics_df["model"], metrics_df["RMSE"], color=["#4C78A8", "#F58518", "#54A24B"])
		ax.set_title("Model RMSE Comparison")
		ax.set_ylabel("RMSE")
		ax.set_xlabel("Model")
		plt.xticks(rotation=20)
		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / "rmse_comparison.png", dpi=150)
		plt.close(fig)

	return metrics_df