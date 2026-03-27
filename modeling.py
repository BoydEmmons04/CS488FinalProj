from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLEANED_DIR = Path("cleaned_data")
OUTPUT_DIR = Path("outputs") / "modeling"


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

	feature_cols = [
		"load_factor",
		"route_avg_load_factor",
		"lag_load_factor",
		"rolling_load_factor_2",
		"market_distance",
		"avg_fuel_price",
		"passengers_db1b",
		"is_saturated",
	]

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

	models = {
		"linear_regression": LinearRegression(),
		"pca_regression": Pipeline(
			[
				("scaler", StandardScaler()),
				("pca", PCA(n_components=0.95)),
				("regressor", LinearRegression()),
			]
		),
		"gradient_boosting": GradientBoostingRegressor(random_state=random_state),
	}

	predictions = {}
	trained_models = {}

	for model_name, model in models.items():
		model.fit(X_train, y_train)
		trained_models[model_name] = model
		predictions[model_name] = model.predict(X_test)

	correlation = model_df[feature_cols + [target_col]].corr(numeric_only=True)

	test_predictions_df = X_test.copy()
	test_predictions_df["actual_avg_fare"] = y_test.values
	for model_name, preds in predictions.items():
		test_predictions_df[f"pred_{model_name}"] = preds

	if save:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

		correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

		fig, ax = plt.subplots(figsize=(10, 8))
		cax = ax.imshow(correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
		ax.set_xticks(range(len(correlation.columns)))
		ax.set_yticks(range(len(correlation.index)))
		ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
		ax.set_yticklabels(correlation.index)
		fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
		ax.set_title("Model Feature Correlation Matrix")
		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
		plt.close(fig)

		test_predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

	return {
		"feature_columns": feature_cols,
		"target_column": target_col,
		"X_test": X_test,
		"y_test": y_test,
		"predictions": predictions,
		"trained_models": trained_models,
		"test_predictions_df": test_predictions_df,
		"correlation": correlation,
	}


def predict_fares(input_df=None, model_name="linear_regression"):
	"""Convenience prediction helper for dashboard/demo use."""
	analysis_df = load_analysis_table() if input_df is None else input_df
	results = run_modeling_pipeline(analysis_df=analysis_df, save=False)
	model = results["trained_models"][model_name]
	preds = model.predict(results["X_test"])

	out_df = results["X_test"].copy()
	out_df["predicted_avg_fare"] = preds
	return out_df