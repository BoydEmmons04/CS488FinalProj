from cleaning import preprocess_all_data
from features import build_analysis_table


def run_pipeline():
    # Runs the currently functional parts of the program
    preprocess_all_data()
    analysis_df = build_analysis_table(save=True)

    print("Pipeline complete.")
    print(f"Analysis table rows: {len(analysis_df):,}")
    print("Saved to cleaned_data/analysis_table.csv")

    return analysis_df


if __name__ == "__main__":
    run_pipeline()
