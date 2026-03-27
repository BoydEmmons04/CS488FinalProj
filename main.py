from cleaning import preprocess_all_data
from features import build_analysis_table
from modeling import run_modeling_pipeline
from evaluation import evaluate_model_outputs


def run_pipeline():
    # Runs the full analytics flow from preprocessing to evaluation.
    preprocess_all_data()
    analysis_df = build_analysis_table(save=True)
    model_results = run_modeling_pipeline(analysis_df=analysis_df, save=True)
    metrics_df = evaluate_model_outputs(model_results, save=True)

    print("Pipeline complete.")
    print(f"Analysis table rows: {len(analysis_df):,}")
    print("Saved to cleaned_data/analysis_table.csv")
    print("Model outputs saved to outputs/modeling/")
    print("Evaluation outputs saved to outputs/evaluation/")

    best = metrics_df.iloc[0]
    print(
        "Best model by RMSE: "
        f"{best['model']} | RMSE={best['RMSE']:.3f} | R2={best['R2']:.3f} | MAPE={best['MAPE']:.3f}"
    )

    return {
        "analysis_df": analysis_df,
        "metrics_df": metrics_df,
    }


if __name__ == "__main__":
    run_pipeline()
