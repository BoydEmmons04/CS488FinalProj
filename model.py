# Filename: model.py
# Purpose:
#   Stage 3 – Feature selection and model frame construction.
#   Stage 4 – Three expert techniques: Correlation Analysis, Linear Regression, PCA Regression.
#   Stage 5 – Performance evaluation metrics: RMSE, MAPE, R², SNR, prediction accuracy.
#   Stage 6 – Data visualization: heatmaps, histograms, scatter plots, accuracy panels.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLEANED_DIR = Path("cleaned_data")
OUTPUT_DIR  = Path("outputs")


# ── SHARED HELPER ──────────────────────────────────────────────────────────────

def _write_table_txt(path, df, title=None, index=False, max_cols_per_block=6):
	# Write a dataframe to a readable plain-text table, split into column blocks
	# so wide tables stay legible in a fixed-width file.
	lines = []
	df_display = df.reset_index() if index else df.copy()
	if title:
		lines.append(title)
		lines.append("=" * len(title))
	lines.append(f"Rows: {len(df_display)} | Columns: {len(df_display.columns)}")
	lines.append("")
	columns = list(df_display.columns)
	for start in range(0, len(columns), max_cols_per_block):
		block = columns[start : start + max_cols_per_block]
		lines.append(f"[Columns {start + 1}-{start + len(block)} of {len(columns)}]")
		lines.append("-" * 80)
		lines.append(df_display[block].to_string(index=False))
		lines.append("")
	path.write_text("\n".join(lines), encoding="utf-8")


# ── STAGE 3: FEATURE SELECTION & MODEL FRAME ──────────────────────────────────

def _build_model_frame(analysis_df):
	# Add temporal load-factor features, impute missing values with column medians,
	# and return the clean dataframe alongside the final feature and target column lists.
	df = analysis_df.copy()
	if "route" not in df.columns:
		df["route"] = df["ORIGIN"] + "-" + df["DEST"]
	# Sort before shifting so each route's lag aligns to the right prior quarter.
	df = df.sort_values(["route", "YEAR", "QUARTER"]).reset_index(drop=True)

	# Route-average and lagged load factor capture capacity context beyond a single quarter.
	df["route_avg_load_factor"] = df.groupby("route")["load_factor"].transform("mean")
	df["lag_load_factor"]       = df.groupby("route")["load_factor"].shift(1)
	df["rolling_load_factor_2"] = df.groupby("route")["load_factor"].transform(
		lambda s: s.rolling(2, min_periods=1).mean()
	)
	# When there is no prior quarter the lag falls back to the current value.
	df["lag_load_factor"]       = df["lag_load_factor"].fillna(df["load_factor"])
	df["rolling_load_factor_2"] = df["rolling_load_factor_2"].fillna(df["load_factor"])

	# Fixed 19-feature set covering capacity, distance, fuel, competition, and delay signals.
	feature_candidates = [
		"load_factor", "route_avg_load_factor", "lag_load_factor", "rolling_load_factor_2",
		"market_distance", "avg_fuel_price", "passengers_db1b",
		"competition_flights", "competition_unique_carriers", "competition_avg_delay",
		"competition_cancel_rate", "competition_delay15_rate",
		"route_avg_arr_delay_rate", "route_avg_cancel_rate",
		"route_weather_delay_share", "route_nas_delay_share",
		"route_late_aircraft_delay_share", "route_carrier_delay_share",
		"is_saturated",
	]
	feature_cols = [c for c in feature_candidates if c in df.columns]

	# Impute each numeric predictor with its median; is_saturated gets 0 for unknowns.
	for col in feature_cols:
		if col == "is_saturated":
			df[col] = df[col].fillna(0).astype(int)
		else:
			median = df[col].median()
			df[col] = df[col].fillna(median if not pd.isna(median) else 0.0)

	target_col = "avg_fare"
	required   = feature_cols + [target_col]
	missing    = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns for modeling: {missing}")

	return df.dropna(subset=required).copy(), feature_cols, target_col


# ── STAGE 4: EXPERT TECHNIQUES ────────────────────────────────────────────────
# Technique 1 – Correlation Analysis   : pairwise Pearson correlations, focus matrix + heatmap.
# Technique 2 – Linear Regression      : OLS on the full 19-feature set.
# Technique 3 – PCA Regression         : StandardScaler → PCA (95% variance) → OLS.

def run_modeling_pipeline(analysis_df=None, test_size=0.2, random_state=42, save=True):
	"""Train all three expert techniques and return results for evaluation."""
	if analysis_df is None:
		analysis_df = pd.read_csv(CLEANED_DIR / "analysis_table.csv")

	model_df, feature_cols, target_col = _build_model_frame(analysis_df)
	X = model_df[feature_cols]
	y = model_df[target_col]

	# Fixed seed keeps the 80/20 train-test split identical across runs.
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state
	)

	# --- Technique 2: Linear Regression ---
	linear_model = LinearRegression()
	linear_model.fit(X_train, y_train)

	# --- Technique 3: PCA Regression ---
	# Features are standardised first so scale differences do not distort PCA components.
	# Components retaining 95% of total variance are kept (typically 9 of 19).
	scaler        = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled  = scaler.transform(X_test)
	pca            = PCA(n_components=0.95, random_state=random_state)
	X_train_pca    = pca.fit_transform(X_train_scaled)
	X_test_pca     = pca.transform(X_test_scaled)
	pca_regression = LinearRegression()
	pca_regression.fit(X_train_pca, y_train)

	predictions = {
		"Linear Regression": linear_model.predict(X_test),
		"PCA Regression":    pca_regression.predict(X_test_pca),
	}
	trained_models = {
		"Linear Regression": linear_model,
		"PCA Regression":    pca_regression,
	}

	# --- Technique 1: Correlation Analysis ---
	# Full matrix feeds the heatmap; focus matrix is saved as a readable text table.
	correlation = model_df[feature_cols + [target_col]].corr(numeric_only=True)
	focus_cols  = [c for c in [
		"load_factor", "competition_unique_carriers",
		"route_avg_arr_delay_rate", "avg_fuel_price", "passengers_db1b", target_col,
	] if c in correlation.columns]
	correlation_focus = correlation[focus_cols].loc[focus_cols].copy()

	# Build a per-sample prediction table for the summary text output.
	test_pred_df = X_test.copy()
	test_pred_df["actual_avg_fare"] = y_test.values
	for name, preds in predictions.items():
		test_pred_df[f"pred_{name.lower().replace(' ', '_')}"] = preds

	if save:
		_save_modeling_artifacts(model_df, target_col, correlation, correlation_focus, test_pred_df)

	return {
		"feature_columns":   feature_cols,
		"target_column":     target_col,
		"model_df":          model_df,
		"analysis_df":       analysis_df,
		"X_test":            X_test,
		"y_test":            y_test,
		"predictions":       predictions,
		"trained_models":    trained_models,
		"pca_model":         pca,
		"X_test_pca":        X_test_pca,
		"test_predictions_df": test_pred_df,
		"correlation":       correlation,
		"correlation_focus": correlation_focus,
	}


def _save_modeling_artifacts(model_df, target_col, correlation, correlation_focus, test_pred_df):
	# Write the correlation focus table, overview panel, and prediction summary.
	modeling_dir = OUTPUT_DIR / "modeling"
	modeling_dir.mkdir(parents=True, exist_ok=True)

	# Technique 1 output: correlation focus matrix as text.
	corr_txt = correlation_focus.round(4).copy()
	corr_txt.index.name = "Feature"
	_write_table_txt(
		modeling_dir / "correlation.txt", corr_txt,
		title="CORRELATION FOCUS MATRIX", index=True,
	)

	# Overview panel: full correlation heatmap + three hypothesis-relevant scatter plots.
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	axes = axes.flatten()

	cax = axes[0].imshow(correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
	axes[0].set_xticks(range(len(correlation.columns)))
	axes[0].set_yticks(range(len(correlation.index)))
	axes[0].set_xticklabels(correlation.columns, rotation=45, ha="right", fontsize=8)
	axes[0].set_yticklabels(correlation.index, fontsize=8)
	axes[0].set_title("Correlation Heatmap")
	fig.colorbar(cax, ax=axes[0], fraction=0.046, pad=0.04)

	axes[1].scatter(model_df["load_factor"], model_df[target_col], alpha=0.35, edgecolors="none")
	axes[1].set_xlabel("Load Factor")
	axes[1].set_ylabel("Average Airfare ($)")
	axes[1].set_title("Load Factor vs Airfare")

	if "competition_unique_carriers" in model_df.columns:
		axes[2].scatter(model_df["competition_unique_carriers"], model_df[target_col], alpha=0.35, edgecolors="none")
		axes[2].set_xlabel("Unique Carriers")
		axes[2].set_ylabel("Average Airfare ($)")
		axes[2].set_title("Competition vs Airfare")
	else:
		axes[2].set_visible(False)

	if "route_avg_arr_delay_rate" in model_df.columns:
		axes[3].scatter(model_df["route_avg_arr_delay_rate"], model_df[target_col], alpha=0.35, edgecolors="none")
		axes[3].set_xlabel("Route Avg Delay Rate")
		axes[3].set_ylabel("Average Airfare ($)")
		axes[3].set_title("Delay Rate vs Airfare")
	else:
		axes[3].set_visible(False)

	plt.tight_layout()
	fig.savefig(modeling_dir / "modeling_overview.png", dpi=150)
	plt.close(fig)

	# Prediction summary: per-model error statistics over the test set.
	pred_export_cols = [
		"actual_avg_fare", "pred_linear_regression", "pred_pca_regression",
		"load_factor", "avg_fuel_price", "passengers_db1b",
		"competition_unique_carriers", "route_avg_arr_delay_rate",
	]
	clean_preds = test_pred_df[[c for c in pred_export_cols if c in test_pred_df.columns]].copy()
	clean_preds.insert(0, "sample_id", range(1, len(clean_preds) + 1))

	summary_rows = []
	for model_key, col_pred in [
		("Linear Regression", "pred_linear_regression"),
		("PCA Regression",    "pred_pca_regression"),
	]:
		if col_pred not in clean_preds.columns:
			continue
		residuals = clean_preds["actual_avg_fare"] - clean_preds[col_pred]
		abs_res   = residuals.abs()
		summary_rows.append({
			"Model":               model_key,
			"Samples":             len(clean_preds),
			"Mean_Actual_Fare":    clean_preds["actual_avg_fare"].mean(),
			"Mean_Predicted_Fare": clean_preds[col_pred].mean(),
			"MAE":                 abs_res.mean(),
			"RMSE":                (residuals ** 2).mean() ** 0.5,
			"Median_AE":           abs_res.median(),
			"P90_AE":              abs_res.quantile(0.90),
			"Max_AE":              abs_res.max(),
			"Mean_Residual":       residuals.mean(),
		})

	if summary_rows:
		_write_table_txt(
			modeling_dir / "predictions_summary.txt",
			pd.DataFrame(summary_rows).round(4),
			title="TEST PREDICTIONS SUMMARY",
		)


# ── STAGE 5: PERFORMANCE EVALUATION ───────────────────────────────────────────

def _compute_metrics(y_true, y_pred):
	# RMSE: average prediction error in dollars.
	# MAPE: scale-free accuracy measure.
	# R²  : proportion of fare variance explained.
	# SNR : ratio of signal power (mean fare²) to noise power (MSE).
	rmse = mean_squared_error(y_true, y_pred) ** 0.5
	mape = mean_absolute_percentage_error(y_true, y_pred)
	r2   = r2_score(y_true, y_pred)
	snr  = (np.mean(y_true) ** 2) / (mean_squared_error(y_true, y_pred) + 1e-8)
	return {"RMSE": rmse, "MAPE": mape, "R2": r2, "SNR": snr}


def _build_metrics_df(predictions, y_test):
	# Compute metrics for every model; best model (lowest RMSE) is always row 0.
	rows = []
	for name, y_pred in predictions.items():
		row = {"model": name}
		row.update(_compute_metrics(y_test, y_pred))
		rows.append(row)
	df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
	# Accuracy = (1 - MAPE) × 100 gives an intuitive percentage interpretation.
	df["AccuracyPct"] = ((1.0 - df["MAPE"]) * 100.0).clip(lower=0.0, upper=100.0)
	df["R2Pct"]       = (df["R2"] * 100.0).clip(lower=0.0, upper=100.0)
	return df[["model", "RMSE", "MAPE", "R2", "SNR", "AccuracyPct", "R2Pct"]].round(4)


# ── STAGE 6: DATA VISUALIZATION ───────────────────────────────────────────────

def _plot_model_comparison(export_df, eval_dir):
	# Bar charts comparing RMSE, SNR, R², MAPE, and accuracy side by side (Figure 2).
	colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756", "#72B7B2"]
	fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
	for idx, (col, hint) in enumerate([
		("RMSE_USD",    "Lower Better"),
		("SNR",         "Higher Better"),
		("R2",          "Higher Better"),
		("MAPE",        "Lower Better"),
		("Accuracy_Pct","Higher Better"),
	]):
		axes[idx].bar(export_df["Model"], export_df[col], color=colors[:len(export_df)])
		axes[idx].set_title(f"{col} ({hint})")
		axes[idx].tick_params(axis="x", rotation=18)
	fig.suptitle("Model Comparison Panel", fontsize=14)
	plt.tight_layout()
	fig.savefig(eval_dir / "metrics_comparison.png", dpi=150)
	plt.close(fig)


def _plot_actual_vs_predicted(predictions, y_test, eval_dir):
	# Scatter of actual vs predicted fare with a perfect-prediction dashed line (Figure 3).
	model_names = list(predictions.keys())
	fig, axes = plt.subplots(1, len(model_names), figsize=(8 * len(model_names), 5))
	if len(model_names) == 1:
		axes = [axes]
	for idx, name in enumerate(model_names):
		y_pred = predictions[name]
		axes[idx].scatter(y_test, y_pred, alpha=0.5, edgecolors="black", linewidths=0.2)
		lo, hi = min(float(y_test.min()), float(y_pred.min())), max(float(y_test.max()), float(y_pred.max()))
		axes[idx].plot([lo, hi], [lo, hi], "k--", linewidth=1)
		axes[idx].set_xlabel("Actual Avg Fare ($)")
		axes[idx].set_ylabel("Predicted Avg Fare ($)")
		axes[idx].set_title(name)
	fig.suptitle("Actual vs Predicted Panel", fontsize=14)
	plt.tight_layout()
	fig.savefig(eval_dir / "actual_vs_predicted.png", dpi=150)
	plt.close(fig)


def _plot_diagnostics(model_df, pca_model, eval_dir):
	# Six-panel diagnostics: fare/load histograms, key scatter plots, PCA curve (Figure 4).
	if model_df is None or model_df.empty:
		return
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	axes = axes.flatten()

	# Fare and load factor distributions reveal the right-skewed fare tail.
	axes[0].hist(model_df["avg_fare"],    bins=35, edgecolor="black", alpha=0.75, color="#4C78A8")
	axes[0].set_title("Distribution: Avg Fare")
	axes[0].set_xlabel("avg_fare"); axes[0].set_ylabel("Frequency")

	axes[1].hist(model_df["load_factor"], bins=35, edgecolor="black", alpha=0.75, color="#54A24B")
	axes[1].set_title("Distribution: Load Factor")
	axes[1].set_xlabel("load_factor"); axes[1].set_ylabel("Frequency")

	# Core hypothesis scatter: load factor vs average fare.
	axes[2].scatter(model_df["load_factor"], model_df["avg_fare"], alpha=0.35, edgecolors="none")
	axes[2].set_title("Load Factor vs Avg Fare")
	axes[2].set_xlabel("load_factor"); axes[2].set_ylabel("avg_fare")

	if "competition_unique_carriers" in model_df.columns:
		axes[3].scatter(model_df["competition_unique_carriers"], model_df["avg_fare"], alpha=0.35, edgecolors="none")
		axes[3].set_title("Competition vs Avg Fare")
		axes[3].set_xlabel("competition_unique_carriers"); axes[3].set_ylabel("avg_fare")
	else:
		axes[3].set_visible(False)

	# Fare and load factor trend over available quarters.
	time_df = (
		model_df.groupby(["YEAR", "QUARTER"], as_index=False)
		.agg(avg_fare=("avg_fare", "mean"), avg_load_factor=("load_factor", "mean"))
		.sort_values(["YEAR", "QUARTER"])
	)
	time_df["period"] = time_df["YEAR"].astype(int).astype(str) + "-Q" + time_df["QUARTER"].astype(int).astype(str)
	idx   = np.arange(len(time_df))
	style = "-o" if len(time_df) > 1 else "o"
	axes[4].plot(idx, time_df["avg_fare"],        style, color="#1f77b4", label="Avg Fare")
	axes[4].plot(idx, time_df["avg_load_factor"],  style, color="#d62728", label="Avg Load Factor")
	axes[4].set_xticks(idx)
	axes[4].set_xticklabels(time_df["period"], rotation=20)
	axes[4].set_title("Fare & Load Factor Over Time")
	axes[4].legend(loc="best", fontsize=8)

	# PCA cumulative variance curve shows how many components are needed.
	if pca_model is not None and hasattr(pca_model, "explained_variance_ratio_"):
		cum_var = np.cumsum(pca_model.explained_variance_ratio_)
		axes[5].plot(range(1, len(cum_var) + 1), cum_var, marker="o")
		axes[5].set_ylim([0, 1.05])
		axes[5].set_title("PCA Cumulative Variance")
		axes[5].set_xlabel("Components"); axes[5].set_ylabel("Cumulative Variance")
	else:
		axes[5].set_visible(False)

	for ax in axes:
		if not ax.has_data():
			ax.set_visible(False)

	fig.suptitle("Data Diagnostics Panel", fontsize=14)
	plt.tight_layout()
	fig.savefig(eval_dir / "diagnostics.png", dpi=150)
	plt.close(fig)


def _save_feature_importance(trained_models, X_test, pca_model, eval_dir):
	# Linear regression coefficients ranked by absolute magnitude (Table 2).
	# PCA explained variance breakdown (Table 3).
	feature_cols = list(X_test.columns)

	if "Linear Regression" in trained_models:
		model = trained_models["Linear Regression"]
		if hasattr(model, "coef_"):
			coef = np.ravel(model.coef_)
			n    = min(len(feature_cols), len(coef))
			coef_df = (
				pd.DataFrame({
					"feature":         feature_cols[:n],
					"coefficient":     coef[:n],
					"abs_coefficient": np.abs(coef[:n]),
				})
				.sort_values("abs_coefficient", ascending=False)
			)
			coef_df.insert(0, "rank", range(1, len(coef_df) + 1))
			_write_table_txt(eval_dir / "linear_importance.txt", coef_df.round(4), title="LINEAR FEATURE IMPORTANCE")

	if pca_model is not None and hasattr(pca_model, "components_"):
		explained_df = pd.DataFrame({
			"component":                   [f"PC{i + 1}" for i in range(pca_model.n_components_)],
			"explained_variance_ratio":    pca_model.explained_variance_ratio_,
			"cumulative_explained_variance": np.cumsum(pca_model.explained_variance_ratio_),
		})
		explained_df.insert(0, "component_rank", range(1, len(explained_df) + 1))
		_write_table_txt(eval_dir / "pca_variance.txt", explained_df.round(4), title="PCA EXPLAINED VARIANCE")


def _save_conclusions(export_df, model_df, trained_models, X_test, pca_model, eval_dir):
	# Compact findings table: best model, top Pearson correlations, linear drivers,
	# explicit hypothesis signals, and PCA components.
	rows = []
	corr = None

	if export_df is not None and not export_df.empty:
		best = export_df.sort_values("RMSE_USD").iloc[0]
		rows.append({
			"section": "Best Model",
			"item":    "Top performer by RMSE",
			"value":   best["Model"],
			"detail":  f"RMSE={best['RMSE_USD']:.3f}, R2={best['R2']:.3f}, MAPE={best['MAPE']:.3f}, AccuracyPct={best['Accuracy_Pct']:.2f}%",
		})

	if model_df is not None and not model_df.empty and "avg_fare" in model_df.columns:
		corr = (
			model_df.select_dtypes(include=[np.number])
			.corr(numeric_only=True)["avg_fare"]
			.drop(labels=["avg_fare"], errors="ignore")
			.dropna()
			.sort_values(key=lambda s: s.abs(), ascending=False)
		)
		for feat, val in corr.head(5).items():
			rows.append({"section": "Top Correlations", "item": feat, "value": f"{val:.3f}", "detail": "Pearson with avg_fare"})
		# Always include the direct hypothesis variable even if it is not top-5.
		if "load_factor" in corr.index:
			rows.append(
				{
					"section": "Hypothesis Signals",
					"item": "load_factor correlation",
					"value": f"{corr['load_factor']:.3f}",
					"detail": "Pearson correlation with avg_fare",
				}
			)

	linear_model = trained_models.get("Linear Regression") if trained_models else None
	if linear_model is not None and hasattr(linear_model, "coef_"):
		coef = np.ravel(linear_model.coef_)
		n    = min(len(X_test.columns), len(coef))
		coef_df = (
			pd.DataFrame({
				"feature":         list(X_test.columns)[:n],
				"coefficient":     coef[:n],
				"abs_coefficient": np.abs(coef[:n]),
			})
			.sort_values("abs_coefficient", ascending=False)
		)
		for _, row in coef_df.head(5).iterrows():
			rows.append({"section": "Linear Drivers", "item": row["feature"], "value": f"{row['coefficient']:.3f}", "detail": "Signed linear coefficient"})
		# Always include load-related coefficients for hypothesis interpretation.
		for hypothesis_feature in ["load_factor", "is_saturated"]:
			if hypothesis_feature in coef_df["feature"].values:
				coef_value = coef_df.loc[coef_df["feature"] == hypothesis_feature, "coefficient"].iloc[0]
				rows.append(
					{
						"section": "Hypothesis Signals",
						"item": f"{hypothesis_feature} coefficient",
						"value": f"{coef_value:.3f}",
						"detail": "Linear coefficient holding other features fixed",
					}
				)

	if pca_model is not None and hasattr(pca_model, "components_"):
		feat_names = list(X_test.columns)
		for i in range(min(pca_model.n_components_, 3)):
			loadings = pd.Series(np.abs(pca_model.components_[i]), index=feat_names)
			top      = loadings.idxmax()
			rows.append({
				"section": "PCA Components",
				"item":    f"PC{i + 1} top feature",
				"value":   top,
				"detail":  f"Explains {pca_model.explained_variance_ratio_[i] * 100:.1f}% variance; loading={pca_model.components_[i][loadings.argmax()]:.3f}",
			})

	if rows:
		_write_table_txt(eval_dir / "conclusions.txt", pd.DataFrame(rows), title="CONCLUSIONS SUMMARY")


# ── PUBLIC EVALUATION ENTRY POINT ─────────────────────────────────────────────

def evaluate_model_outputs(model_results, save=True):
	"""Stages 5 & 6: Compute all evaluation metrics and save visualization artifacts."""
	y_test         = model_results["y_test"]
	predictions    = model_results["predictions"]
	trained_models = model_results["trained_models"]
	pca_model      = model_results.get("pca_model")
	X_test         = model_results["X_test"]
	model_df       = model_results.get("model_df")

	metrics_df = _build_metrics_df(predictions, y_test)

	# Rename columns for display in output files and plots.
	export_df = metrics_df.rename(columns={
		"model":       "Model",
		"RMSE":        "RMSE_USD",
		"AccuracyPct": "Accuracy_Pct",
		"R2Pct":       "R2_Pct",
	})

	if save:
		eval_dir = OUTPUT_DIR / "evaluation"
		eval_dir.mkdir(parents=True, exist_ok=True)

		_write_table_txt(eval_dir / "metrics.txt", export_df, title="MODEL METRICS")
		_plot_model_comparison(export_df, eval_dir)
		_plot_actual_vs_predicted(predictions, y_test, eval_dir)
		_plot_diagnostics(model_df, pca_model, eval_dir)
		_save_feature_importance(trained_models, X_test, pca_model, eval_dir)
		_save_conclusions(export_df, model_df, trained_models, X_test, pca_model, eval_dir)

		print("Evaluation artifacts saved to:", eval_dir)

	return metrics_df
