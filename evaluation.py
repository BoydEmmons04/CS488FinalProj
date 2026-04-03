# Filename: evaluation.py
# Purpose: Compare model predictions, compute performance metrics, and export evaluation visuals/tables.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


OUTPUT_DIR = Path("outputs") / "evaluation"


def _write_table_txt(path, df, title=None, index=False, max_cols_per_block=6):
	# Save a dataframe as a table.
	lines = []
	df_display = df.reset_index() if index else df.copy()

	if title:
		lines.append(title)
		lines.append("=" * len(title))

	lines.append(f"Rows: {len(df_display)} | Columns: {len(df_display.columns)}")
	lines.append("")

	columns = list(df_display.columns)
	for start in range(0, len(columns), max_cols_per_block):
		block_cols = columns[start : start + max_cols_per_block]
		block_df = df_display[block_cols]
		lines.append(f"[Columns {start + 1}-{start + len(block_cols)} of {len(columns)}]")
		lines.append("-" * 80)
		lines.append(block_df.to_string(index=False))
		lines.append("")

	path.write_text("\n".join(lines), encoding="utf-8")


def _save_report_table_txt(metrics_df, model_df, trained_models, X_test, pca_model=None):
	# Write a short summary for the report.
	lines = []
	lines.append("AIRLINE FARE MODEL REPORT SUMMARY")
	lines.append("=" * 80)
	lines.append("")

	lines.append("[1] MODEL METRICS")
	lines.append("-" * 80)
	if metrics_df is not None and not metrics_df.empty:
		lines.append(metrics_df.to_string(index=False))
	else:
		lines.append("No model metrics available.")
	lines.append("")

	top_corr_df = pd.DataFrame()
	if model_df is not None and not model_df.empty and "avg_fare" in model_df.columns:
		numeric_df = model_df.select_dtypes(include=[np.number]).copy()
		if "avg_fare" in numeric_df.columns:
			corr = numeric_df.corr(numeric_only=True)["avg_fare"].drop(labels=["avg_fare"], errors="ignore")
			corr = corr.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
			top_corr_df = corr.head(10).reset_index()
			top_corr_df.columns = ["Feature", "Corr_with_AvgFare"]
			top_corr_df["Corr_with_AvgFare"] = top_corr_df["Corr_with_AvgFare"].round(4)

	lines.append("[2] TOP CORRELATIONS WITH AVERAGE FARE")
	lines.append("-" * 80)
	if not top_corr_df.empty:
		lines.append(top_corr_df.to_string(index=False))
	else:
		lines.append("No correlation summary available.")
	lines.append("")

	linear_driver_df = pd.DataFrame()
	linear_model = trained_models.get("Linear Regression") if trained_models else None
	if linear_model is not None and hasattr(linear_model, "coef_"):
		coef = np.ravel(linear_model.coef_)
		n = min(len(X_test.columns), len(coef))
		linear_driver_df = pd.DataFrame(
			{
				"Feature": list(X_test.columns)[:n],
				"Coefficient": coef[:n],
				"Abs_Coefficient": np.abs(coef[:n]),
			}
		).sort_values("Abs_Coefficient", ascending=False)
		linear_driver_df.insert(0, "Rank", range(1, len(linear_driver_df) + 1))
		linear_driver_df = linear_driver_df.head(10).round(4)

	lines.append("[3] TOP LINEAR REGRESSION DRIVERS")
	lines.append("-" * 80)
	if not linear_driver_df.empty:
		lines.append(linear_driver_df.to_string(index=False))
	else:
		lines.append("No linear-regression driver summary available.")
	lines.append("")

	pca_table_df = pd.DataFrame()
	if pca_model is not None and hasattr(pca_model, "explained_variance_ratio_"):
		pca_table_df = pd.DataFrame(
			{
				"Component": [f"PC{i + 1}" for i in range(pca_model.n_components_)],
				"Explained_Variance": pca_model.explained_variance_ratio_,
				"Cumulative_Variance": np.cumsum(pca_model.explained_variance_ratio_),
			}
		)
		pca_table_df.insert(0, "Rank", range(1, len(pca_table_df) + 1))
		pca_table_df = pca_table_df.head(10).round(4)

	lines.append("[4] PCA EXPLAINED VARIANCE")
	lines.append("-" * 80)
	if not pca_table_df.empty:
		lines.append(pca_table_df.to_string(index=False))
	else:
		lines.append("No PCA variance summary available.")
	lines.append("")

	best_model_line = "No best model available."
	if metrics_df is not None and not metrics_df.empty:
		best = metrics_df.sort_values("RMSE_USD").iloc[0]
		best_model_line = (
			f"Best model by RMSE: {best['Model']} | RMSE={best['RMSE_USD']:.4f} | "
			f"R2={best['R2']:.4f} | MAPE={best['MAPE']:.4f} | Accuracy={best['Accuracy_Pct']:.2f}%"
		)

	lines.append("[5] KEY TAKEAWAY")
	lines.append("-" * 80)
	lines.append(best_model_line)
	lines.append("")

	(OUTPUT_DIR / "report_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _compute_metrics(y_true, y_pred):
	# Core regression metrics.
	rmse = mean_squared_error(y_true, y_pred) ** 0.5
	mape = mean_absolute_percentage_error(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	signal_power = np.mean(y_true) ** 2
	noise_power = mean_squared_error(y_true, y_pred)
	snr = signal_power / (noise_power + 1e-8)
	
	return {
		"RMSE": rmse,
		"MAPE": mape,
		"R2": r2,
		"SNR": snr,
	}


def _add_accuracy_columns(metrics_df):
	# Add percent-style accuracy fields.
	metrics_df["AccuracyPct"] = ((1.0 - metrics_df["MAPE"]) * 100.0).clip(lower=0.0, upper=100.0)
	metrics_df["R2Pct"] = (metrics_df["R2"] * 100.0).clip(lower=0.0, upper=100.0)
	return metrics_df


def _save_grouped_visuals(metrics_df, predictions, y_test, model_df, pca_model=None):
	# Save the main comparison and diagnostic plots.
	base_colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756", "#72B7B2"]
	colors = [base_colors[i % len(base_colors)] for i in range(len(metrics_df))]

	fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
	metric_specs = [
		("RMSE_USD", "Lower Better"),
		("SNR", "Higher Better"),
		("R2", "Higher Better"),
		("MAPE", "Lower Better"),
		("Accuracy_Pct", "Higher Better"),
	]
	for idx, (metric_col, hint) in enumerate(metric_specs):
		axes[idx].bar(metrics_df["Model"], metrics_df[metric_col], color=colors)
		axes[idx].set_title(f"{metric_col} ({hint})")
		axes[idx].tick_params(axis="x", rotation=18)
	fig.suptitle("Model Comparison Panel", fontsize=14)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=150)
	plt.close(fig)

	model_names = list(predictions.keys())
	fig, axes = plt.subplots(1, len(model_names), figsize=(8 * len(model_names), 5))
	if len(model_names) == 1:
		axes = [axes]
	for idx, model_name in enumerate(model_names):
		y_pred = predictions[model_name]
		axes[idx].scatter(y_test, y_pred, alpha=0.5, edgecolors="black", linewidths=0.2)
		min_val = min(float(y_test.min()), float(y_pred.min()))
		max_val = max(float(y_test.max()), float(y_pred.max()))
		# Dashed line shows a perfect prediction.
		axes[idx].plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
		axes[idx].set_xlabel("Actual Avg Fare ($)")
		axes[idx].set_ylabel("Predicted Avg Fare ($)")
		axes[idx].set_title(model_name)
	fig.suptitle("Actual vs Predicted Panel", fontsize=14)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
	plt.close(fig)

	if model_df is None or model_df.empty:
		return

	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	axes = axes.flatten()

	if "avg_fare" in model_df.columns:
		axes[0].hist(model_df["avg_fare"], bins=35, edgecolor="black", alpha=0.75, color="#4C78A8")
		axes[0].set_title("Distribution: avg_fare")
		axes[0].set_xlabel("avg_fare")
		axes[0].set_ylabel("Frequency")

	if "load_factor" in model_df.columns:
		axes[1].hist(model_df["load_factor"], bins=35, edgecolor="black", alpha=0.75, color="#54A24B")
		axes[1].set_title("Distribution: load_factor")
		axes[1].set_xlabel("load_factor")
		axes[1].set_ylabel("Frequency")

	if "load_factor" in model_df.columns and "avg_fare" in model_df.columns:
		axes[2].scatter(model_df["load_factor"], model_df["avg_fare"], alpha=0.35, edgecolors="none")
		axes[2].set_title("Load Factor vs Avg Fare")
		axes[2].set_xlabel("load_factor")
		axes[2].set_ylabel("avg_fare")

	if "competition_unique_carriers" in model_df.columns and "avg_fare" in model_df.columns:
		axes[3].scatter(
			model_df["competition_unique_carriers"],
			model_df["avg_fare"],
			alpha=0.35,
			edgecolors="none",
		)
		axes[3].set_title("Competition vs Avg Fare")
		axes[3].set_xlabel("competition_unique_carriers")
		axes[3].set_ylabel("avg_fare")
	elif "route_avg_arr_delay_rate" in model_df.columns and "avg_fare" in model_df.columns:
		axes[3].scatter(
			model_df["route_avg_arr_delay_rate"],
			model_df["avg_fare"],
			alpha=0.35,
			edgecolors="none",
		)
		axes[3].set_title("Delay Rate vs Avg Fare")
		axes[3].set_xlabel("route_avg_arr_delay_rate")
		axes[3].set_ylabel("avg_fare")

	time_df = (
		model_df.groupby(["YEAR", "QUARTER"], as_index=False)
		.agg(avg_fare=("avg_fare", "mean"), avg_load_factor=("load_factor", "mean"), avg_fuel_price=("avg_fuel_price", "mean"))
		.sort_values(["YEAR", "QUARTER"])
	)
	# Use YYYY-QN labels on the x-axis.
	time_df["period"] = (
		time_df["YEAR"].astype(int).astype(str)
		+ "-Q"
		+ time_df["QUARTER"].astype(int).astype(str)
	)
	_write_table_txt(
		OUTPUT_DIR / "time_series.txt",
		time_df.round(4),
		title="TIME SERIES SUMMARY",
		index=False,
	)
	period_idx = np.arange(len(time_df))
	line_style = "-o" if len(time_df) > 1 else "o"
	axes[4].plot(period_idx, time_df["avg_fare"], line_style, color="#1f77b4", label="Avg Fare")
	axes[4].plot(period_idx, time_df["avg_load_factor"], line_style, color="#d62728", label="Avg Load")
	axes[4].set_xticks(period_idx)
	axes[4].set_xticklabels(time_df["period"], rotation=20)
	axes[4].set_title("Fare & Load Over Time")
	axes[4].legend(loc="best", fontsize=8)

	if pca_model is not None and hasattr(pca_model, "explained_variance_ratio_"):
		cum_var = np.cumsum(pca_model.explained_variance_ratio_)
		axes[5].plot(range(1, len(cum_var) + 1), cum_var, marker="o")
		axes[5].set_ylim([0, 1.05])
		axes[5].set_title("PCA Cumulative Variance")
		axes[5].set_xlabel("Components")
		axes[5].set_ylabel("Cumulative Variance")
	else:
		axes[5].set_visible(False)

	for ax in axes:
		if not ax.has_data():
			ax.set_visible(False)

	fig.suptitle("Data Diagnostics Panel", fontsize=14)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / "diagnostics.png", dpi=150)
	plt.close(fig)


def _save_feature_importance_artifacts(trained_models, X_test, pca_model=None):
	# Save linear coefficients and PCA variance tables.
	feature_cols = list(X_test.columns)

	if "Linear Regression" in trained_models:
		linear_model = trained_models["Linear Regression"]
		if hasattr(linear_model, "coef_"):
			coef = np.ravel(linear_model.coef_)
			n = min(len(feature_cols), len(coef))
			coef_df = pd.DataFrame(
				{
					"feature": feature_cols[:n],
					"coefficient": coef[:n],
					"abs_coefficient": np.abs(coef[:n]),
				}
			# Sort by absolute value so the biggest drivers show up first regardless of sign.
			).sort_values("abs_coefficient", ascending=False)
			coef_df.insert(0, "rank", range(1, len(coef_df) + 1))
			coef_df = coef_df.round(4)
			_write_table_txt(
				OUTPUT_DIR / "linear_importance.txt",
				coef_df,
				title="LINEAR FEATURE IMPORTANCE",
				index=False,
			)

	if pca_model is not None and hasattr(pca_model, "components_"):
		component_cols = [f"PC{i + 1}" for i in range(pca_model.n_components_)]
		explained_df = pd.DataFrame(
			{
				"component": component_cols,
				"explained_variance_ratio": pca_model.explained_variance_ratio_,
				"cumulative_explained_variance": np.cumsum(pca_model.explained_variance_ratio_),
			}
		)
		explained_df.insert(0, "component_rank", range(1, len(explained_df) + 1))
		explained_df = explained_df.round(4)
		_write_table_txt(
			OUTPUT_DIR / "pca_variance.txt",
			explained_df,
			title="PCA EXPLAINED VARIANCE",
			index=False,
		)


def _save_conclusions_summary(metrics_df, model_df, trained_models, X_test, pca_model=None):
	# Save a short findings table.
	rows = []

	if metrics_df is not None and not metrics_df.empty:
		best = metrics_df.sort_values("RMSE_USD").iloc[0]
		rows.append(
			{
				"section": "Best Model",
				"item": "Top performer by RMSE",
				"value": best["Model"],
				"detail": (
					f"RMSE={best['RMSE_USD']:.3f}, R2={best['R2']:.3f}, "
					f"MAPE={best['MAPE']:.3f}, AccuracyPct={best['Accuracy_Pct']:.2f}%"
				),
			}
		)

	if model_df is not None and not model_df.empty and "avg_fare" in model_df.columns:
		numeric_df = model_df.select_dtypes(include=[np.number]).copy()
		if "avg_fare" in numeric_df.columns:
			corr = numeric_df.corr(numeric_only=True)["avg_fare"].drop(labels=["avg_fare"], errors="ignore")
			corr = corr.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
			for feature_name, corr_value in corr.head(5).items():
				rows.append(
					{
						"section": "Top Correlations",
						"item": feature_name,
						"value": f"{corr_value:.3f}",
						"detail": "Pearson correlation with avg_fare",
					}
				)

	linear_model = trained_models.get("Linear Regression") if trained_models else None
	if linear_model is not None and hasattr(linear_model, "coef_"):
		coef_df = pd.DataFrame(
			{
				"feature": list(X_test.columns),
				"coefficient": linear_model.coef_,
				"abs_coefficient": np.abs(linear_model.coef_),
			}
		).sort_values("abs_coefficient", ascending=False)
		for _, row in coef_df.head(5).iterrows():
			rows.append(
				{
					"section": "Linear Drivers",
					"item": row["feature"],
					"value": f"{row['coefficient']:.3f}",
					"detail": "Linear coefficient (signed impact)",
				}
			)

	if pca_model is not None and hasattr(pca_model, "components_") and hasattr(pca_model, "explained_variance_ratio_"):
		feature_names = list(X_test.columns)
		for i in range(min(pca_model.n_components_, 3)):
			loadings = pd.Series(np.abs(pca_model.components_[i]), index=feature_names)
			top_feature = loadings.idxmax()
			rows.append(
				{
					"section": "PCA Components",
					"item": f"PC{i+1} top feature",
					"value": top_feature,
					"detail": (
						f"Explains {pca_model.explained_variance_ratio_[i]*100:.1f}% variance; "
						f"loading={pca_model.components_[i][loadings.argmax()]:.3f}"
					),
				}
			)

	if rows:
		_write_table_txt(
			OUTPUT_DIR / "conclusions.txt",
			pd.DataFrame(rows),
			title="CONCLUSIONS SUMMARY",
			index=False,
		)


def evaluate_model_outputs(model_results, save=True):
	# Run evaluation and save outputs.
	y_test = model_results["y_test"]
	predictions = model_results["predictions"]
	trained_models = model_results["trained_models"]
	pca_model = model_results.get("pca_model")
	X_test = model_results["X_test"]
	model_df = model_results.get("model_df")

	metric_rows = []
	for model_name, y_pred in predictions.items():
		row = {"model": model_name}
		row.update(_compute_metrics(y_test, y_pred))
		metric_rows.append(row)

	# Sort by RMSE so the best model is always row 0.
	metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
	metrics_df = _add_accuracy_columns(metrics_df)
	metrics_df = metrics_df[["model", "RMSE", "MAPE", "R2", "SNR", "AccuracyPct", "R2Pct"]].round(4)
	metrics_df = metrics_df.rename(
		columns={
			"model": "Model",
			"RMSE": "RMSE_USD",
			"MAPE": "MAPE",
			"R2": "R2",
			"SNR": "SNR",
			"AccuracyPct": "Accuracy_Pct",
			"R2Pct": "R2_Pct",
		}
	)

	if save:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

		_write_table_txt(
			OUTPUT_DIR / "metrics.txt",
			metrics_df,
			title="MODEL METRICS",
			index=False,
		)
		_save_grouped_visuals(metrics_df, predictions, y_test, model_df, pca_model=pca_model)
		_save_feature_importance_artifacts(trained_models, X_test, pca_model=pca_model)
		_save_conclusions_summary(metrics_df, model_df, trained_models, X_test, pca_model=pca_model)
		_save_report_table_txt(metrics_df, model_df, trained_models, X_test, pca_model=pca_model)

		print("Cleaned evaluation artifacts saved to:", OUTPUT_DIR)

	return metrics_df