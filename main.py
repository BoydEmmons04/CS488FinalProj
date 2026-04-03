# Filename: main.py
# Purpose: Run the project pipeline end to end.

from ingest import preprocess_all_data
from features import build_analysis_table
from modeling import run_modeling_pipeline
from evaluation import evaluate_model_outputs


def run_pipeline():
    # Run each stage in order.
    preprocess_all_data()
    analysis_df = build_analysis_table(save=True)
    model_results = run_modeling_pipeline(analysis_df=analysis_df, save=True)
    metrics_df = evaluate_model_outputs(model_results, save=True)

    print("Pipeline complete.")
    print(f"Analysis table rows: {len(analysis_df):,}")

    best = metrics_df.iloc[0]
    accuracy_text = ""
    if "AccuracyPct" in metrics_df.columns:
        accuracy_text = f" | Accuracy={best['AccuracyPct']:.2f}%"
    print(
        "Best model by RMSE: "
        f"{best['model']} | RMSE={best['RMSE']:.3f} | R2={best['R2']:.3f} | MAPE={best['MAPE']:.3f}{accuracy_text}"
    )


if __name__ == "__main__":
    run_pipeline()
