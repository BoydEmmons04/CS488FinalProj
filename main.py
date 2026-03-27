from cleaning import preprocess_all_data
from features import (
    build_analysis_table,
    select_features_regression,
    select_features_classification,
    apply_pca,
    ALL_NUMERIC_FEATURES,
)
from modeling import run_all_models, load_analysis_table
from evaluation import run_all_evaluation


def run_pipeline():
    # ── Step 1: Data Preprocessing ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Data Preprocessing")
    print("=" * 60)
    preprocess_all_data()

    # ── Step 2: Feature Engineering ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering – Build Analysis Table")
    print("=" * 60)
    analysis_df = build_analysis_table(save=True)
    print(f"Analysis table rows: {len(analysis_df):,}")
    print("Saved to cleaned_data/analysis_table.csv")

    # ── Step 3: Feature Selection & Dimensionality Reduction ─────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Feature Selection / PCA")
    print("=" * 60)
    clean_df = analysis_df.dropna(subset=ALL_NUMERIC_FEATURES)

    reg_selected, reg_scores = select_features_regression(clean_df, k=5)
    print(f"\nTop 5 regression features (SelectKBest): {reg_selected}")

    clf_selected, clf_scores = select_features_classification(clean_df, k=5)
    print(f"Top 5 classification features (SelectKBest): {clf_selected}")

    pca_obj, pca_df, pca_explained = apply_pca(clean_df, n_components=3)
    print(f"PCA explained variance: {pca_explained.to_dict()}")
    print(f"Total variance explained by 3 PCs: {pca_explained.sum():.4f}")

    # ── Step 4: Modeling – Three Expert Techniques ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Modeling – Three Expert Techniques")
    print("=" * 60)
    model_df = load_analysis_table()
    results = run_all_models()

    # ── Step 5 & 6: Evaluation Metrics & Visualizations ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 & 6: Performance Evaluation & Data Visualization")
    print("=" * 60)
    run_all_evaluation(
        results,
        model_df,
        reg_scores=reg_scores,
        clf_scores=clf_scores,
        pca_explained=pca_explained,
    )

    print("\nPipeline complete.")
    return analysis_df


if __name__ == "__main__":
    run_pipeline()

