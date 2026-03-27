"""
Builds the merged feature table used for airfare analysis and modeling.

This module loads the cleaned airline and fuel datasets, aggregates them to a
shared route and quarter level, and creates the core variables used in later
modeling steps, including load factor and saturation indicators.
"""


from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import StandardScaler


CLEANED_DIR = Path("cleaned_data")


def _safe_divide(numerator, denominator):
    """Divide stuff without crashing on zero denominator"""
    denominator = denominator.where(denominator != 0)
    return numerator / denominator


def load_cleaned_inputs():
    """Load the cleaned files from /cleaned_data"""
    return {
        "db1b": pd.read_csv(CLEANED_DIR / "db1b_cleaned.csv"),
        "t100": pd.read_csv(CLEANED_DIR / "t100_cleaned.csv"),
        "fuel": pd.read_csv(CLEANED_DIR / "fuel_cleaned.csv"),
    }


def aggregate_db1b_routes(db1b_df):
    """Compress DB1B down to one row per route and quarter."""
    return (
        db1b_df.groupby(["YEAR", "QUARTER", "ORIGIN", "DEST"], as_index=False)
        .agg(
            avg_fare=("MARKET_FARE", "mean"),
            passengers_db1b=("PASSENGERS", "sum"),
            market_distance=("MARKET_DISTANCE", "mean"),
        )
    )


def aggregate_t100_routes(t100_df):
    """Compress T-100 data into route-level flight stats."""
    return (
        t100_df.groupby(["YEAR", "QUARTER", "ORIGIN", "DEST"], as_index=False)
        .agg(
            passengers_t100=("PASSENGERS", "sum"),
            seats=("SEATS", "sum"),
            departures_performed=("DEPARTURES_PERFORMED", "sum"),
        )
    )


def aggregate_fuel_quarterly(fuel_df):
    """Turn daily fuel prices into one quarter average."""
    fuel_df = fuel_df.copy()
    fuel_df["observation_date"] = pd.to_datetime(fuel_df["observation_date"])
    fuel_df["YEAR"] = fuel_df["observation_date"].dt.year
    fuel_df["QUARTER"] = fuel_df["observation_date"].dt.quarter

    return (
        fuel_df.groupby(["YEAR", "QUARTER"], as_index=False)
        .agg(avg_fuel_price=("DJFUELUSGULF", "mean"))
    )


def build_analysis_table(save=True):
    """Create the modeling table."""
    datasets = load_cleaned_inputs()

    db1b_routes = aggregate_db1b_routes(datasets["db1b"])
    t100_routes = aggregate_t100_routes(datasets["t100"])
    fuel_quarterly = aggregate_fuel_quarterly(datasets["fuel"])

    analysis_df = db1b_routes.merge(
        t100_routes,
        on=["YEAR", "QUARTER", "ORIGIN", "DEST"],
        how="inner",
    ).merge(
        fuel_quarterly,
        on=["YEAR", "QUARTER"],
        how="left",
    )

    analysis_df["load_factor"] = _safe_divide(
        analysis_df["passengers_t100"], analysis_df["seats"]
    )
    analysis_df["route"] = analysis_df["ORIGIN"] + "-" + analysis_df["DEST"]
    analysis_df["is_saturated"] = (analysis_df["load_factor"] >= 0.8).astype(int)

    analysis_df = analysis_df.sort_values(
        ["YEAR", "QUARTER", "ORIGIN", "DEST"]
    ).reset_index(drop=True)

    if save:
        CLEANED_DIR.mkdir(exist_ok=True)
        analysis_df.to_csv(CLEANED_DIR / "analysis_table.csv", index=False)

    return analysis_df


# ─── Feature Selection & Dimensionality Reduction ────────────────────────────

# Numeric feature columns available in the analysis table
ALL_NUMERIC_FEATURES = [
    "avg_fare",
    "passengers_db1b",
    "market_distance",
    "passengers_t100",
    "seats",
    "departures_performed",
    "avg_fuel_price",
    "load_factor",
]


def drop_highly_correlated(df, features, threshold=0.95):
    """Return a reduced feature list by removing highly correlated pairs.

    For any pair of features with |r| >= threshold the second feature in the
    pair is dropped, keeping the one that appears first.
    """
    corr_matrix = df[features].corr().abs()
    to_drop = set()
    for i, col_i in enumerate(features):
        for col_j in features[i + 1 :]:
            if col_i not in to_drop and corr_matrix.loc[col_i, col_j] >= threshold:
                to_drop.add(col_j)
    return [f for f in features if f not in to_drop]


def select_features_regression(df, target="avg_fare", k=5):
    """Use SelectKBest (F-regression) to choose the k best predictors.

    Returns
    -------
    selected : list[str]   – ordered list of selected feature names
    scores   : pd.Series   – F-scores for all candidate features
    """
    candidates = [f for f in ALL_NUMERIC_FEATURES if f != target]
    X = df[candidates].dropna()
    y = df.loc[X.index, target]

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=candidates).sort_values(ascending=False)
    selected = scores.head(k).index.tolist()
    return selected, scores


def select_features_classification(df, target="is_saturated", k=5):
    """Use SelectKBest (F-classif / ANOVA) to choose the k best predictors.

    Returns
    -------
    selected : list[str]   – ordered list of selected feature names
    scores   : pd.Series   – F-scores for all candidate features
    """
    candidates = [f for f in ALL_NUMERIC_FEATURES if f != target]
    X = df[candidates].dropna()
    y = df.loc[X.index, target]

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=candidates).sort_values(ascending=False)
    selected = scores.head(k).index.tolist()
    return selected, scores


def apply_pca(df, features=None, n_components=3):
    """Apply PCA to the numeric features of the analysis table.

    Parameters
    ----------
    df          : pd.DataFrame  – analysis table (rows must have no NaNs in features)
    features    : list[str]     – columns to reduce (defaults to ALL_NUMERIC_FEATURES)
    n_components: int           – number of principal components to keep

    Returns
    -------
    pca         : fitted sklearn PCA object
    pca_df      : pd.DataFrame with columns PC1, PC2, … PCn plus non-numeric columns
    explained   : pd.Series     – explained variance ratio per component
    """
    if features is None:
        features = ALL_NUMERIC_FEATURES

    sub = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)

    col_names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=col_names, index=sub.index)

    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=col_names,
        name="explained_variance_ratio",
    )
    return pca, pca_df, explained


if __name__ == "__main__":
    analysis_df = build_analysis_table(save=True)
    print(
        f"Analysis table created with {len(analysis_df):,} rows "
        f"and saved to {CLEANED_DIR / 'analysis_table.csv'}."
    )

    clean_df = analysis_df.dropna(subset=ALL_NUMERIC_FEATURES)

    print("\n--- Correlation-based feature reduction (threshold=0.95) ---")
    reduced = drop_highly_correlated(clean_df, ALL_NUMERIC_FEATURES)
    print(f"  Kept {len(reduced)}/{len(ALL_NUMERIC_FEATURES)} features: {reduced}")

    print("\n--- SelectKBest for regression (target=avg_fare, k=5) ---")
    reg_sel, reg_scores = select_features_regression(clean_df, k=5)
    print(f"  Selected: {reg_sel}")
    print(f"  F-scores:\n{reg_scores.to_string()}")

    print("\n--- SelectKBest for classification (target=is_saturated, k=5) ---")
    clf_sel, clf_scores = select_features_classification(clean_df, k=5)
    print(f"  Selected: {clf_sel}")
    print(f"  F-scores:\n{clf_scores.to_string()}")

    print("\n--- PCA (3 components) ---")
    pca_obj, pca_df, explained = apply_pca(clean_df, n_components=3)
    print(f"  Explained variance ratio: {explained.to_dict()}")
    print(f"  Total variance explained: {explained.sum():.4f}")
