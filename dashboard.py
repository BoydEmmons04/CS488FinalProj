"""
Shows modeling results, metrics, predictions, and artifacts for correlation analysis, linear regression, and PCA.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(".")
METRICS_PATH = ROOT / "outputs" / "evaluation" / "model_metrics.csv"
PREDS_PATH = ROOT / "outputs" / "modeling" / "test_predictions.csv"
EVAL_DIR = ROOT / "outputs" / "evaluation"
MODEL_DIR = ROOT / "outputs" / "modeling"


def load_outputs():
	metrics_df = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None
	preds_df = pd.read_csv(PREDS_PATH) if PREDS_PATH.exists() else None
	return metrics_df, preds_df


def main():
	st.title("Airline Fare Modeling Dashboard")
	st.caption("Core techniques: Correlation analysis, Linear Regression, and PCA")

	metrics_df, preds_df = load_outputs()
	if metrics_df is None or preds_df is None:
		st.warning("No model outputs found yet. Run: python main.py")
		return

	tab_metrics, tab_predictions, tab_visuals, tab_features = st.tabs(
		["Metrics", "Predictions", "Visuals", "Feature Importance"]
	)

	with tab_metrics:
		st.subheader("Model Metrics")
		st.dataframe(metrics_df, use_container_width=True)

		if "AccuracyPct" in metrics_df.columns:
			best_acc_idx = metrics_df["AccuracyPct"].idxmax()
			best_acc_model = metrics_df.loc[best_acc_idx, "model"]
			best_acc_value = metrics_df.loc[best_acc_idx, "AccuracyPct"]
			st.metric("Best Accuracy (100 - MAPE)", f"{best_acc_value:.2f}%", delta=best_acc_model)

		col1, col2, col3 = st.columns(3)
		with col1:
			fig, ax = plt.subplots(figsize=(7, 4))
			ax.bar(metrics_df["model"], metrics_df["RMSE"], color="#4C78A8")
			ax.set_ylabel("RMSE")
			ax.set_xlabel("Model")
			ax.set_title("RMSE by Model")
			plt.xticks(rotation=20)
			st.pyplot(fig)

		with col2:
			if "SNR" in metrics_df.columns:
				fig, ax = plt.subplots(figsize=(7, 4))
				ax.bar(metrics_df["model"], metrics_df["SNR"], color="#54A24B")
				ax.set_ylabel("SNR")
				ax.set_xlabel("Model")
				ax.set_title("SNR by Model")
				plt.xticks(rotation=20)
				st.pyplot(fig)

		with col3:
			if "AccuracyPct" in metrics_df.columns:
				fig, ax = plt.subplots(figsize=(7, 4))
				ax.bar(metrics_df["model"], metrics_df["AccuracyPct"], color="#2E8B57")
				ax.set_ylabel("Accuracy (%)")
				ax.set_xlabel("Model")
				ax.set_title("Accuracy by Model")
				ax.set_ylim([0, 100])
				plt.xticks(rotation=20)
				st.pyplot(fig)

	with tab_predictions:
		st.subheader("Prediction Table")
		model_pred_cols = [c for c in preds_df.columns if c.startswith("pred_")]
		selected_cols = st.multiselect(
			"Select prediction columns",
			options=model_pred_cols,
			default=model_pred_cols,
		)
		display_cols = [
			c
			for c in ["actual_avg_fare", *selected_cols, "load_factor", "avg_fuel_price"]
			if c in preds_df.columns
		]
		st.dataframe(preds_df[display_cols].head(200), use_container_width=True)

	with tab_visuals:
		st.subheader("Saved Modeling and Evaluation Visuals")
		image_files = [
			MODEL_DIR / "correlation_heatmap.png",
			MODEL_DIR / "load_factor_vs_airfare.png",
			MODEL_DIR / "competition_vs_airfare.png",
			MODEL_DIR / "delay_rate_vs_airfare.png",
			MODEL_DIR / "correlation_focus.csv",
			EVAL_DIR / "rmse_snr_comparison.png",
			EVAL_DIR / "r2_mape_comparison.png",
			EVAL_DIR / "accuracy_comparison.png",
			EVAL_DIR / "key_variable_histograms.png",
			EVAL_DIR / "feature_relationships_vs_fare.png",
			EVAL_DIR / "delay_cause_composition.png",
			EVAL_DIR / "pca_explained_variance.png",
			EVAL_DIR / "fare_load_factor_over_time.png",
			EVAL_DIR / "fuel_price_over_time.png",
		]

		for image_path in image_files:
			if image_path.suffix.lower() == ".png" and image_path.exists():
				st.image(str(image_path), caption=image_path.name, use_container_width=True)
			if image_path.suffix.lower() == ".csv" and image_path.exists():
				st.markdown(f"**{image_path.name}**")
				st.dataframe(pd.read_csv(image_path), use_container_width=True)

		for model_name in metrics_df["model"]:
			file_slug = model_name.lower().replace(" ", "_")
			scatter_path = EVAL_DIR / f"actual_vs_predicted_{file_slug}.png"
			if scatter_path.exists():
				st.image(str(scatter_path), caption=scatter_path.name, use_container_width=True)

	with tab_features:
		st.subheader("Linear and PCA Artifacts")
		importance_files = {
			"Linear Coefficients": EVAL_DIR / "feature_importance_linear.csv",
			"PCA Explained Variance": EVAL_DIR / "pca_explained_variance.csv",
			"Time Summary": EVAL_DIR / "time_series_summary.csv",
			"Conclusions Summary": EVAL_DIR / "conclusions_summary.csv",
		}

		for label, file_path in importance_files.items():
			if file_path.exists():
				st.markdown(f"**{label}**")
				st.dataframe(pd.read_csv(file_path), use_container_width=True)


if __name__ == "__main__":
	main()