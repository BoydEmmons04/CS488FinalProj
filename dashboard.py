from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(".")
METRICS_PATH = ROOT / "outputs" / "evaluation" / "model_metrics.csv"
PREDS_PATH = ROOT / "outputs" / "modeling" / "test_predictions.csv"


def load_outputs():
	metrics_df = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None
	preds_df = pd.read_csv(PREDS_PATH) if PREDS_PATH.exists() else None
	return metrics_df, preds_df


def main():
	st.title("Airline Fare Modeling Dashboard")
	st.caption("Early-stage dashboard for model performance and prediction outputs")

	metrics_df, preds_df = load_outputs()
	if metrics_df is None or preds_df is None:
		st.warning("No model outputs found yet. Run: python main.py")
		return

	st.subheader("Model Metrics")
	st.dataframe(metrics_df, use_container_width=True)

	st.subheader("RMSE Comparison")
	fig, ax = plt.subplots(figsize=(7, 4))
	ax.bar(metrics_df["model"], metrics_df["RMSE"])
	ax.set_ylabel("RMSE")
	ax.set_xlabel("Model")
	ax.set_title("RMSE by Model")
	plt.xticks(rotation=20)
	st.pyplot(fig)

	st.subheader("Prediction Sample")
	st.dataframe(preds_df.head(50), use_container_width=True)


if __name__ == "__main__":
	main()