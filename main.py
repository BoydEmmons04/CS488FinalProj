# Filename: main.py
# Purpose: Orchestrate the full pipeline in stage order.

from pathlib import Path

from data  import preprocess_all_data, build_analysis_table
from model import run_modeling_pipeline, evaluate_model_outputs


OUTPUT_DIR = Path("outputs")


def ensure_output_dirs_exist():
    # Ensure output folders are present.
    (OUTPUT_DIR / "modeling").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "evaluation").mkdir(parents=True, exist_ok=True)


def clear_output_files_for_run():
    # Regenerate outputs each run by clearing only existing files.
    for subdir in [OUTPUT_DIR / "modeling", OUTPUT_DIR / "evaluation"]:
        if not subdir.exists():
            continue
        for file_path in subdir.glob("*"):
            if file_path.is_file():
                file_path.unlink()


def run_pipeline():
    ensure_output_dirs_exist()
    clear_output_files_for_run()

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
