"""
Builds the merged route‑by‑quarter feature table from cleaned airline and fuel data, including load‑factor and saturation variables.
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
        "competition": pd.read_csv(CLEANED_DIR / "competition_cleaned.csv"),
        "delay": pd.read_csv(CLEANED_DIR / "delay_cleaned.csv"),
        "db1b": pd.read_csv(CLEANED_DIR / "db1b_cleaned.csv"),
        "t100": pd.read_csv(CLEANED_DIR / "t100_cleaned.csv"),
        "fuel": pd.read_csv(CLEANED_DIR / "fuel_cleaned.csv"),
    }


def aggregate_competition_routes(comp_df):
    """Compress competition flights into route-level quarterly pressure signals."""
    comp_df = comp_df.copy()
    comp_df["Date"] = pd.to_datetime(comp_df["Date"])
    comp_df["YEAR"] = comp_df["Date"].dt.year
    comp_df["QUARTER"] = comp_df["Date"].dt.quarter
    comp_df["Delay"] = pd.to_numeric(comp_df["Delay"], errors="coerce")
    comp_df["Cancelled"] = pd.to_numeric(comp_df["Cancelled"], errors="coerce").fillna(0)
    comp_df["is_delayed_15"] = (comp_df["Delay"] >= 15).astype(int)

    return (
        comp_df.groupby(["YEAR", "QUARTER", "Origin", "Dest"], as_index=False)
        .agg(
            competition_flights=("Carrier", "size"),
            competition_unique_carriers=("Carrier", "nunique"),
            competition_avg_delay=("Delay", "mean"),
            competition_cancel_rate=("Cancelled", "mean"),
            competition_delay15_rate=("is_delayed_15", "mean"),
        )
        .rename(columns={"Origin": "ORIGIN", "Dest": "DEST"})
    )


def aggregate_delay_airport_quarter(delay_df):
    """Create airport-quarter delay severity features from BTS delay-cause data."""
    delay_df = delay_df.copy()
    delay_df["QUARTER"] = ((delay_df["month"] - 1) // 3) + 1

    agg_df = (
        delay_df.groupby(["year", "QUARTER", "airport"], as_index=False)
        .agg(
            arr_flights_total=("arr_flights", "sum"),
            arr_del15_total=("arr_del15", "sum"),
            arr_cancelled_total=("arr_cancelled", "sum"),
            weather_delay_total=("weather_delay", "sum"),
            nas_delay_total=("nas_delay", "sum"),
            late_aircraft_delay_total=("late_aircraft_delay", "sum"),
            carrier_delay_total=("carrier_delay", "sum"),
            arr_delay_total=("arr_delay", "sum"),
        )
        .rename(columns={"year": "YEAR"})
    )

    agg_df["airport_arr_delay_rate"] = _safe_divide(
        agg_df["arr_del15_total"], agg_df["arr_flights_total"]
    )
    agg_df["airport_cancel_rate"] = _safe_divide(
        agg_df["arr_cancelled_total"], agg_df["arr_flights_total"]
    )
    agg_df["weather_delay_share"] = _safe_divide(
        agg_df["weather_delay_total"], agg_df["arr_delay_total"]
    )
    agg_df["nas_delay_share"] = _safe_divide(
        agg_df["nas_delay_total"], agg_df["arr_delay_total"]
    )
    agg_df["late_aircraft_delay_share"] = _safe_divide(
        agg_df["late_aircraft_delay_total"], agg_df["arr_delay_total"]
    )
    agg_df["carrier_delay_share"] = _safe_divide(
        agg_df["carrier_delay_total"], agg_df["arr_delay_total"]
    )

    keep_cols = [
        "YEAR",
        "QUARTER",
        "airport",
        "airport_arr_delay_rate",
        "airport_cancel_rate",
        "weather_delay_share",
        "nas_delay_share",
        "late_aircraft_delay_share",
        "carrier_delay_share",
    ]
    return agg_df[keep_cols]


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

    competition_routes = aggregate_competition_routes(datasets["competition"])
    delay_airport = aggregate_delay_airport_quarter(datasets["delay"])
    db1b_routes = aggregate_db1b_routes(datasets["db1b"])
    t100_routes = aggregate_t100_routes(datasets["t100"])
    fuel_quarterly = aggregate_fuel_quarterly(datasets["fuel"])

    analysis_df = db1b_routes.merge(
        t100_routes,
        on=["YEAR", "QUARTER", "ORIGIN", "DEST"],
        how="inner",
    ).merge(
        competition_routes,
        on=["YEAR", "QUARTER", "ORIGIN", "DEST"],
        how="left",
    ).merge(
        fuel_quarterly,
        on=["YEAR", "QUARTER"],
        how="left",
    )

    origin_delay = delay_airport.rename(
        columns={
            "airport": "ORIGIN",
            "airport_arr_delay_rate": "origin_arr_delay_rate",
            "airport_cancel_rate": "origin_cancel_rate",
            "weather_delay_share": "origin_weather_delay_share",
            "nas_delay_share": "origin_nas_delay_share",
            "late_aircraft_delay_share": "origin_late_aircraft_delay_share",
            "carrier_delay_share": "origin_carrier_delay_share",
        }
    )
    dest_delay = delay_airport.rename(
        columns={
            "airport": "DEST",
            "airport_arr_delay_rate": "dest_arr_delay_rate",
            "airport_cancel_rate": "dest_cancel_rate",
            "weather_delay_share": "dest_weather_delay_share",
            "nas_delay_share": "dest_nas_delay_share",
            "late_aircraft_delay_share": "dest_late_aircraft_delay_share",
            "carrier_delay_share": "dest_carrier_delay_share",
        }
    )

    analysis_df = analysis_df.merge(
        origin_delay,
        on=["YEAR", "QUARTER", "ORIGIN"],
        how="left",
    ).merge(
        dest_delay,
        on=["YEAR", "QUARTER", "DEST"],
        how="left",
    )

    analysis_df["load_factor"] = _safe_divide(
        analysis_df["passengers_t100"], analysis_df["seats"]
    )
    analysis_df["route_avg_arr_delay_rate"] = (
        analysis_df[["origin_arr_delay_rate", "dest_arr_delay_rate"]].mean(axis=1)
    )
    analysis_df["route_avg_cancel_rate"] = (
        analysis_df[["origin_cancel_rate", "dest_cancel_rate"]].mean(axis=1)
    )
    analysis_df["route_weather_delay_share"] = (
        analysis_df[["origin_weather_delay_share", "dest_weather_delay_share"]].mean(axis=1)
    )
    analysis_df["route_nas_delay_share"] = (
        analysis_df[["origin_nas_delay_share", "dest_nas_delay_share"]].mean(axis=1)
    )
    analysis_df["route_late_aircraft_delay_share"] = (
        analysis_df[
            ["origin_late_aircraft_delay_share", "dest_late_aircraft_delay_share"]
        ].mean(axis=1)
    )
    analysis_df["route_carrier_delay_share"] = (
        analysis_df[["origin_carrier_delay_share", "dest_carrier_delay_share"]].mean(axis=1)
    )

    for col in [
        "competition_flights",
        "competition_unique_carriers",
        "competition_avg_delay",
        "competition_cancel_rate",
        "competition_delay15_rate",
    ]:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].fillna(0)

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
    build_analysis_table(save=True)
