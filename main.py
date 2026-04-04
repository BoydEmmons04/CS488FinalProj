# Filename: main.py
# Purpose: Run the full pipeline in stage order.

import shutil
from pathlib import Path

from data  import preprocess_all_data, build_analysis_table
from model import run_modeling_pipeline, evaluate_model_outputs


OUTPUT_DIR = Path("outputs")
TMP_OUTPUT_DIR = Path("_outputs_tmp")


def prepare_temp_output_dirs():
    # Build outputs in a temp folder during execution.
    if TMP_OUTPUT_DIR.exists():
        shutil.rmtree(TMP_OUTPUT_DIR)
    (TMP_OUTPUT_DIR / "modeling").mkdir(parents=True, exist_ok=True)
    (TMP_OUTPUT_DIR / "evaluation").mkdir(parents=True, exist_ok=True)


def publish_outputs():
    # Publish temp outputs to the final outputs folder at the end.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subdir in [OUTPUT_DIR / "modeling", OUTPUT_DIR / "evaluation"]:
        subdir.mkdir(parents=True, exist_ok=True)
        for file_path in subdir.glob("*"):
            if file_path.is_file():
                file_path.unlink()

    for name in ["modeling", "evaluation"]:
        src_dir = TMP_OUTPUT_DIR / name
        dst_dir = OUTPUT_DIR / name
        if not src_dir.exists():
            continue
        for file_path in src_dir.glob("*"):
            if file_path.is_file():
                file_path.replace(dst_dir / file_path.name)

    shutil.rmtree(TMP_OUTPUT_DIR, ignore_errors=True)


def run_pipeline():
    prepare_temp_output_dirs()

    # Stage 1 & 2: load raw data, clean it, and build the route-quarter feature table.
    preprocess_all_data()
    analysis_df = build_analysis_table(save=True)

    # Stage 3 & 4: select features, run Correlation Analysis, Linear Regression, PCA Regression.
    model_results = run_modeling_pipeline(analysis_df=analysis_df, save=True, output_root=TMP_OUTPUT_DIR)

    # Stage 5 & 6: compute RMSE/MAPE/R²/SNR/Accuracy metrics and save all visualizations.
    metrics_df = evaluate_model_outputs(model_results, save=True, output_root=TMP_OUTPUT_DIR)

    publish_outputs()

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
