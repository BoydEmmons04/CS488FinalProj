"""
Builds the merged feature table used for airfare analysis and modeling.

This module loads the cleaned airline and fuel datasets, aggregates them to a
shared route and quarter level, and creates the core variables used in later
modeling steps, including load factor and saturation indicators.
"""


from pathlib import Path

import pandas as pd


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


if __name__ == "__main__":
    analysis_df = build_analysis_table(save=True)
    print(
        f"Analysis table created with {len(analysis_df):,} rows "
        f"and saved to {CLEANED_DIR / 'analysis_table.csv'}."
    )
