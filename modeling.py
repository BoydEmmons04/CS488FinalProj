"""
Trains regression and dimensionality reduction models, generates predictions, and produces correlation analyses and visualizations.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CLEANED_DIR = Path("cleaned_data")
OUTPUT_DIR = Path("outputs") / "modeling"


def _write_table_txt(path, df, title=None, index=False, max_cols_per_block=6):
	"""Write a dataframe to a readable plain-text table file."""
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

def load_analysis_table(path=CLEANED_DIR / "analysis_table.csv"):
	"""Load the merged analysis table used for modeling."""
	return pd.read_csv(path)


def _build_model_frame(analysis_df):
	"""Create a stable model frame with derived features and no nulls in predictors."""
	df = analysis_df.copy()

	if "route" not in df.columns:
		df["route"] = df["ORIGIN"] + "-" + df["DEST"]

	df = df.sort_values(["route", "YEAR", "QUARTER"]).reset_index(drop=True)

	# Route context features that align with the project hypothesis.
	df["route_avg_load_factor"] = df.groupby("route")["load_factor"].transform("mean")
	df["lag_load_factor"] = df.groupby("route")["load_factor"].shift(1)
	df["rolling_load_factor_2"] = (
		df.groupby("route")["load_factor"].transform(lambda s: s.rolling(2, min_periods=1).mean())
	)

	if "lag_load_factor" in df.columns:
		df["lag_load_factor"] = df["lag_load_factor"].fillna(df["load_factor"])
	if "rolling_load_factor_2" in df.columns:
		df["rolling_load_factor_2"] = df["rolling_load_factor_2"].fillna(df["load_factor"])

	numeric_cols = [
		"load_factor",
		"route_avg_load_factor",
		"lag_load_factor",
		"rolling_load_factor_2",
		"market_distance",
		"avg_fuel_price",
		"passengers_db1b",
		"competition_flights",
		"competition_unique_carriers",
		"competition_avg_delay",
		"competition_cancel_rate",
		"competition_delay15_rate",
		"route_avg_arr_delay_rate",
		"route_avg_cancel_rate",
		"route_weather_delay_share",
		"route_nas_delay_share",
		"route_late_aircraft_delay_share",
		"route_carrier_delay_share",
	]
	for col in numeric_cols:
		if col not in df.columns:
			continue
		median = df[col].median()
		if pd.isna(median):
			median = 0.0
		df[col] = df[col].fillna(median)
	if "is_saturated" in df.columns:
		df["is_saturated"] = df["is_saturated"].fillna(0).astype(int)

	feature_candidates = [
		"load_factor",
		"route_avg_load_factor",
		"lag_load_factor",
		"rolling_load_factor_2",
		"market_distance",
		"avg_fuel_price",
		"passengers_db1b",
		"competition_flights",
		"competition_unique_carriers",
		"competition_avg_delay",
		"competition_cancel_rate",
		"competition_delay15_rate",
		"route_avg_arr_delay_rate",
		"route_avg_cancel_rate",
		"route_weather_delay_share",
		"route_nas_delay_share",
		"route_late_aircraft_delay_share",
		"route_carrier_delay_share",
		"is_saturated",
	]
	feature_cols = [c for c in feature_candidates if c in df.columns]

	target_col = "avg_fare"
	required_cols = feature_cols + [target_col]
	missing_cols = [c for c in required_cols if c not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required columns for modeling: {missing_cols}")

	model_df = df.dropna(subset=required_cols).copy()
	return model_df, feature_cols, target_col


def run_modeling_pipeline(analysis_df=None, test_size=0.2, random_state=42, save=True):
	"""Train project models and return prediction artifacts for evaluation."""
	if analysis_df is None:
		analysis_df = load_analysis_table()

	model_df, feature_cols, target_col = _build_model_frame(analysis_df)

	X = model_df[feature_cols]
	y = model_df[target_col]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state
	)

	linear_model = LinearRegression()
	linear_model.fit(X_train, y_train)

	# Core model 2: PCA + Regression — reduce to 95% variance components then regress.
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	pca = PCA(n_components=0.95, random_state=random_state)
	X_train_pca = pca.fit_transform(X_train_scaled)
	X_test_pca = pca.transform(X_test_scaled)

	pca_regression = LinearRegression()
	pca_regression.fit(X_train_pca, y_train)

	predictions = {
		"Linear Regression": linear_model.predict(X_test),
		"PCA Regression": pca_regression.predict(X_test_pca),
	}
	trained_models = {
		"Linear Regression": linear_model,
		"PCA Regression": pca_regression,
	}

	correlation = model_df[feature_cols + [target_col]].corr(numeric_only=True)
	focus_cols = [
		"load_factor",
		"competition_unique_carriers",
		"route_avg_arr_delay_rate",
		"avg_fuel_price",
		"passengers_db1b",
		target_col,
	]
	focus_cols = [c for c in focus_cols if c in correlation.columns]
	correlation_focus = correlation[focus_cols].loc[focus_cols].copy()

	test_predictions_df = X_test.copy()
	test_predictions_df["actual_avg_fare"] = y_test.values
	for model_name, preds in predictions.items():
		pred_col = model_name.lower().replace(" ", "_")
		test_predictions_df[f"pred_{pred_col}"] = preds

	if save:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

		correlation_focus_txt = correlation_focus.round(4).copy()
		correlation_focus_txt.index.name = "Feature"
		_write_table_txt(
			OUTPUT_DIR / "correlation_focus.txt",
			correlation_focus_txt,
			title="CORRELATION FOCUS MATRIX",
			index=True,
		)

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
			axes[2].scatter(
				model_df["competition_unique_carriers"],
				model_df[target_col],
				alpha=0.35,
				edgecolors="none",
			)
			axes[2].set_xlabel("Unique Carriers")
			axes[2].set_ylabel("Average Airfare ($)")
			axes[2].set_title("Competition vs Airfare")
		else:
			axes[2].set_visible(False)

		if "route_avg_arr_delay_rate" in model_df.columns:
			axes[3].scatter(
				model_df["route_avg_arr_delay_rate"],
				model_df[target_col],
				alpha=0.35,
				edgecolors="none",
			)
			axes[3].set_xlabel("Route Avg Delay Rate")
			axes[3].set_ylabel("Average Airfare ($)")
			axes[3].set_title("Delay Rate vs Airfare")
		else:
			axes[3].set_visible(False)

		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / "modeling_overview_panel.png", dpi=150)
		plt.close(fig)

		pred_export_cols = [
			"actual_avg_fare",
			"pred_linear_regression",
			"pred_pca_regression",
			"load_factor",
			"avg_fuel_price",
			"passengers_db1b",
			"competition_unique_carriers",
			"route_avg_arr_delay_rate",
		]
		existing_cols = [c for c in pred_export_cols if c in test_predictions_df.columns]
		clean_preds = test_predictions_df[existing_cols].copy()
		clean_preds.insert(0, "sample_id", range(1, len(clean_preds) + 1))
		if "pred_linear_regression" in clean_preds.columns:
			clean_preds["residual_linear_regression"] = (
				clean_preds["actual_avg_fare"] - clean_preds["pred_linear_regression"]
			)
		if "pred_pca_regression" in clean_preds.columns:
			clean_preds["residual_pca_regression"] = (
				clean_preds["actual_avg_fare"] - clean_preds["pred_pca_regression"]
			)
		clean_preds = clean_preds.round(4)

		pred_summary_rows = []
		if "residual_linear_regression" in clean_preds.columns:
			lin_abs = clean_preds["residual_linear_regression"].abs()
			pred_summary_rows.append(
				{
					"Model": "Linear Regression",
					"Samples": len(clean_preds),
					"Mean_Actual_Fare": clean_preds["actual_avg_fare"].mean(),
					"Mean_Predicted_Fare": clean_preds["pred_linear_regression"].mean(),
					"MAE": lin_abs.mean(),
					"RMSE": (clean_preds["residual_linear_regression"] ** 2).mean() ** 0.5,
					"Median_AE": lin_abs.median(),
					"P90_AE": lin_abs.quantile(0.90),
					"Max_AE": lin_abs.max(),
					"Mean_Residual": clean_preds["residual_linear_regression"].mean(),
				}
			)
		if "residual_pca_regression" in clean_preds.columns:
			pca_abs = clean_preds["residual_pca_regression"].abs()
			pred_summary_rows.append(
				{
					"Model": "PCA Regression",
					"Samples": len(clean_preds),
					"Mean_Actual_Fare": clean_preds["actual_avg_fare"].mean(),
					"Mean_Predicted_Fare": clean_preds["pred_pca_regression"].mean(),
					"MAE": pca_abs.mean(),
					"RMSE": (clean_preds["residual_pca_regression"] ** 2).mean() ** 0.5,
					"Median_AE": pca_abs.median(),
					"P90_AE": pca_abs.quantile(0.90),
					"Max_AE": pca_abs.max(),
					"Mean_Residual": clean_preds["residual_pca_regression"].mean(),
				}
			)
		if pred_summary_rows:
			pred_summary_df = pd.DataFrame(pred_summary_rows).round(4)
			_write_table_txt(
				OUTPUT_DIR / "test_predictions_summary.txt",
				pred_summary_df,
				title="TEST PREDICTIONS SUMMARY",
				index=False,
			)

	return {
		"feature_columns": feature_cols,
		"target_column": target_col,
		"model_df": model_df,
		"analysis_df": analysis_df,
		"X_test": X_test,
		"y_test": y_test,
		"predictions": predictions,
		"trained_models": trained_models,
		"pca_model": pca,
		"X_test_pca": X_test_pca,
		"test_predictions_df": test_predictions_df,
		"correlation": correlation,
		"correlation_focus": correlation_focus,
	}


def predict_fares(input_df=None, model_name="Linear Regression"):
	"""Generate predictions on input data using trained models."""
	analysis_df = load_analysis_table() if input_df is None else input_df
	results = run_modeling_pipeline(analysis_df=analysis_df, save=False)
	if model_name == "linear_regression":
		model_name = "Linear Regression"
	model = results["trained_models"][model_name]
	X_input = results["X_test_pca"] if model_name == "PCA Regression" else results["X_test"]
	out_df = results["X_test"].copy()
	out_df["predicted_avg_fare"] = model.predict(X_input)
	return out_df