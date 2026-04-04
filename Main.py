# Filename: main.py
# Purpose: Run the full pipeline in stage order.

from pathlib import Path

from Data  import preprocess_all_data, build_analysis_table
from Model import run_modeling_pipeline, evaluate_model_outputs


OUTPUT_DIR = Path("Outputs")


def create_output_dirs():
    # Create output folders.
    (OUTPUT_DIR / "Modeling").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "Evaluation").mkdir(parents=True, exist_ok=True)


def run_pipeline():
    create_output_dirs()

    # Stage 1 & 2: load raw data, clean it, and build the route-quarter feature table.
    preprocess_all_data()
    analysis_df = build_analysis_table(save=True)

    # Stage 3 & 4: select features, run Correlation Analysis, Linear Regression, PCA Regression.
    model_results = run_modeling_pipeline(analysis_df=analysis_df, save=True)

    # Stage 5 & 6: compute RMSE/MAPE/R²/SNR/Accuracy metrics and save all visualizations.
    metrics_df = evaluate_model_outputs(model_results, save=True)

    print("Pipeline complete.")
    print(f"Analysis table rows: {len(analysis_df):,}")

    best = metrics_df.iloc[0]
    accuracy_text = f" | Accuracy={best['AccuracyPct']:.2f}%" if "AccuracyPct" in metrics_df.columns else ""
    print(
        f"Best model by RMSE: {best['model']} | "
        f"RMSE={best['RMSE']:.3f} | R2={best['R2']:.3f} | MAPE={best['MAPE']:.3f}{accuracy_text}"
    )


if __name__ == "__main__":
    run_pipeline()
