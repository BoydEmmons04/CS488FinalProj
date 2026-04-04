# Filename: data.py
# Purpose:
#   Stage 1 – Load raw CSVs, filter to Q1 2025 and target airports, write cleaned tables.
#   Stage 2 – Merge cleaned sources and build the route-quarter feature table for modeling.

from pathlib import Path

import pandas as pd

# ── PATHS & SCOPE ──────────────────────────────────────────────────────────────

DATA_DIR    = Path("Project Datasets")
CLEANED_DIR = Path("cleaned_data")

CA_AIRPORTS = ["LAX", "SFO", "SAN", "OAK", "ONT", "BUR", "SJC", "SMF", "PSP", "FAT", "BFL"]
GA_AIRPORTS = ["ATL", "SAV", "AGS", "ABY"]
TX_AIRPORTS = ["DFW", "IAH", "DAL", "HOU", "AUS", "SAT", "ELP", "MAF", "ABI", "AMA", "GRK"]
TARGET_AIRPORTS = set(CA_AIRPORTS + GA_AIRPORTS + TX_AIRPORTS)


# ── STAGE 1: DATA PREPROCESSING ───────────────────────────────────────────────

def _load_competition():
	# Competition flight file is large, so we read and filter in chunks.
	file_path = DATA_DIR / "Competition (Airline Count)" / "US Flights Data (2025, Q1) - Flight Dataset.csv"
	chunks = []
	for chunk in pd.read_csv(file_path, chunksize=100_000):
		chunk["Date"] = pd.to_datetime(chunk["Date"])
		chunk = chunk[(chunk["Date"] >= "2025-01-01") & (chunk["Date"] <= "2025-03-31")]
		chunk = chunk[chunk["Origin"].isin(TARGET_AIRPORTS) | chunk["Dest"].isin(TARGET_AIRPORTS)]
		if not chunk.empty:
			chunks.append(chunk)
	df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
	if not df.empty:
		df["Delay"]     = pd.to_numeric(df["Delay"], errors="coerce")
		df["Cancelled"] = df["Cancelled"].astype(int)
	return df


def _load_db1b(state):
	# DB1B ticket data: keep only Q1 2025 rows at target airports.
	file_path = DATA_DIR / "DB1BMarket Airline Ticket Data" / f"{state}_T_DB1B_MARKET.csv"
	cols = ["YEAR", "QUARTER", "ORIGIN", "DEST", "PASSENGERS", "MARKET_FARE", "MARKET_DISTANCE"]
	chunks = []
	for chunk in pd.read_csv(file_path, usecols=cols, chunksize=100_000):
		chunk = chunk[(chunk["YEAR"] == 2025) & (chunk["QUARTER"] == 1)]
		chunk = chunk[chunk["ORIGIN"].isin(TARGET_AIRPORTS) | chunk["DEST"].isin(TARGET_AIRPORTS)]
		if not chunk.empty:
			chunks.append(chunk)
	df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
	for col in ["PASSENGERS", "MARKET_FARE", "MARKET_DISTANCE"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def _load_delay():
	# BTS delay cause data: Q1 2025 rows at target airports.
	# Some numeric columns are stored as text in the raw file.
	file_path = DATA_DIR / "Flight Delays" / "ot_delaycause1_DL" / "Airline_Delay_Cause.csv"
	df = pd.read_csv(file_path)
	df = df[(df["year"] == 2025) & (df["month"].isin([1, 2, 3]))]
	df = df[df["airport"].isin(TARGET_AIRPORTS)]
	numeric_cols = [
		"arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct",
		"security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted",
		"arr_delay", "carrier_delay", "weather_delay", "nas_delay",
		"security_delay", "late_aircraft_delay",
	]
	for col in numeric_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def _load_fuel():
	# Gulf Coast jet fuel spot prices filtered to Q1 2025.
	file_path = DATA_DIR / "Fuel Prices" / "DJFUELUSGULF.csv"
	df = pd.read_csv(file_path)
	df["observation_date"] = pd.to_datetime(df["observation_date"])
	df = df[(df["observation_date"] >= "2025-01-01") & (df["observation_date"] <= "2025-03-31")]
	df["DJFUELUSGULF"] = pd.to_numeric(df["DJFUELUSGULF"], errors="coerce")
	return df


def _load_t100(state):
	# T-100 segment stats for one state, Q1 2025 only.
	file_path = DATA_DIR / "T-100 (Load Factor)" / f"{state}_T_T100D_SEGMENT_US_CARRIER_ONLY.csv"
	df = pd.read_csv(file_path)
	df = df[(df["YEAR"] == 2025) & (df["QUARTER"] == 1)]
	for col in ["DEPARTURES_PERFORMED", "SEATS", "PASSENGERS"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def preprocess_all_data(output_dir=CLEANED_DIR):
	"""Stage 1: Load all five raw sources, scope them to Q1 2025, and write cleaned CSVs."""
	print("Loading competition data...")
	competition_df = _load_competition()

	print("Loading delay data...")
	delay_df = _load_delay()

	print("Loading DB1B data...")
	db1b_df = pd.concat(
		[_load_db1b("CA"), _load_db1b("GA"), _load_db1b("TX")],
		ignore_index=True,
	)

	print("Loading fuel data...")
	fuel_df = _load_fuel()

	print("Loading T-100 data...")
	t100_df = pd.concat(
		[_load_t100("CA"), _load_t100("GA"), _load_t100("TX")],
		ignore_index=True,
	)

	output_dir.mkdir(parents=True, exist_ok=True)
	competition_df.to_csv(output_dir / "competition_cleaned.csv", index=False)
	delay_df.to_csv(output_dir / "delay_cleaned.csv",             index=False)
	db1b_df.to_csv(output_dir / "db1b_cleaned.csv",               index=False)
	fuel_df.to_csv(output_dir / "fuel_cleaned.csv",               index=False)
	t100_df.to_csv(output_dir / "t100_cleaned.csv",               index=False)

	print("Preprocessing complete. Cleaned datasets saved to", output_dir)


# ── STAGE 2: FEATURE ENGINEERING & ANALYSIS TABLE ─────────────────────────────

def _safe_divide(numerator, denominator):
	# Zero denominators become NaN rather than inf.
	return numerator / denominator.where(denominator != 0)


def _agg_competition(comp_df):
	# Summarize flight counts, carrier count, and delay signals by route and quarter.
	comp_df = comp_df.copy()
	comp_df["Date"]        = pd.to_datetime(comp_df["Date"])
	comp_df["YEAR"]        = comp_df["Date"].dt.year
	comp_df["QUARTER"]     = comp_df["Date"].dt.quarter
	comp_df["Delay"]       = pd.to_numeric(comp_df["Delay"], errors="coerce")
	comp_df["Cancelled"]   = pd.to_numeric(comp_df["Cancelled"], errors="coerce").fillna(0)
	# BTS standard: a flight is delayed if it arrives 15+ minutes late.
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


def _agg_delay(delay_df):
	# Compute airport-level arrival delay and cancellation rates by quarter.
	# The raw file uses month; convert to quarter for the join key.
	delay_df = delay_df.copy()
	delay_df["QUARTER"] = ((delay_df["month"] - 1) // 3) + 1
	agg = (
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
	agg["airport_arr_delay_rate"]      = _safe_divide(agg["arr_del15_total"],              agg["arr_flights_total"])
	agg["airport_cancel_rate"]         = _safe_divide(agg["arr_cancelled_total"],          agg["arr_flights_total"])
	agg["weather_delay_share"]         = _safe_divide(agg["weather_delay_total"],          agg["arr_delay_total"])
	agg["nas_delay_share"]             = _safe_divide(agg["nas_delay_total"],              agg["arr_delay_total"])
	agg["late_aircraft_delay_share"]   = _safe_divide(agg["late_aircraft_delay_total"],    agg["arr_delay_total"])
	agg["carrier_delay_share"]         = _safe_divide(agg["carrier_delay_total"],          agg["arr_delay_total"])
	return agg[[
		"YEAR", "QUARTER", "airport",
		"airport_arr_delay_rate", "airport_cancel_rate",
		"weather_delay_share", "nas_delay_share",
		"late_aircraft_delay_share", "carrier_delay_share",
	]]


def _agg_db1b(db1b_df):
	# Average fare, total passengers, and mean distance per route-quarter.
	return (
		db1b_df.groupby(["YEAR", "QUARTER", "ORIGIN", "DEST"], as_index=False)
		.agg(
			avg_fare=("MARKET_FARE", "mean"),
			passengers_db1b=("PASSENGERS", "sum"),
			market_distance=("MARKET_DISTANCE", "mean"),
		)
	)


def _agg_t100(t100_df):
	# Total passengers, seats, and departures per route-quarter (needed to compute load factor).
	return (
		t100_df.groupby(["YEAR", "QUARTER", "ORIGIN", "DEST"], as_index=False)
		.agg(
			passengers_t100=("PASSENGERS", "sum"),
			seats=("SEATS", "sum"),
			departures_performed=("DEPARTURES_PERFORMED", "sum"),
		)
	)


def _agg_fuel(fuel_df):
	# Quarterly average jet fuel price.
	fuel_df = fuel_df.copy()
	fuel_df["observation_date"] = pd.to_datetime(fuel_df["observation_date"])
	fuel_df["YEAR"]    = fuel_df["observation_date"].dt.year
	fuel_df["QUARTER"] = fuel_df["observation_date"].dt.quarter
	return fuel_df.groupby(["YEAR", "QUARTER"], as_index=False).agg(avg_fuel_price=("DJFUELUSGULF", "mean"))


def build_analysis_table(save=True):
	"""Stage 2: Merge all cleaned sources into the route-quarter analysis table.

	Join order:
	  DB1B (fares + demand) INNER T-100 (capacity) → ensures only routes with
	  both fare and traffic data are kept, then LEFT join competition, fuel,
	  and delay features so missing auxiliary data doesn't drop valid routes.
	"""
	competition = pd.read_csv(CLEANED_DIR / "competition_cleaned.csv")
	delay       = pd.read_csv(CLEANED_DIR / "delay_cleaned.csv")
	db1b        = pd.read_csv(CLEANED_DIR / "db1b_cleaned.csv")
	t100        = pd.read_csv(CLEANED_DIR / "t100_cleaned.csv")
	fuel        = pd.read_csv(CLEANED_DIR / "fuel_cleaned.csv")

	db1b_routes  = _agg_db1b(db1b)
	t100_routes  = _agg_t100(t100)
	comp_routes  = _agg_competition(competition)
	delay_airport = _agg_delay(delay)
	fuel_quarterly = _agg_fuel(fuel)

	df = (
		db1b_routes
		.merge(t100_routes,  on=["YEAR", "QUARTER", "ORIGIN", "DEST"], how="inner")
		.merge(comp_routes,  on=["YEAR", "QUARTER", "ORIGIN", "DEST"], how="left")
		.merge(fuel_quarterly, on=["YEAR", "QUARTER"],                 how="left")
	)

	# Attach delay severity features for both the origin and destination airports,
	# then average them into a single route-level value for each delay category.
	origin_delay = delay_airport.rename(columns={
		"airport":                   "ORIGIN",
		"airport_arr_delay_rate":    "origin_arr_delay_rate",
		"airport_cancel_rate":       "origin_cancel_rate",
		"weather_delay_share":       "origin_weather_delay_share",
		"nas_delay_share":           "origin_nas_delay_share",
		"late_aircraft_delay_share": "origin_late_aircraft_delay_share",
		"carrier_delay_share":       "origin_carrier_delay_share",
	})
	dest_delay = delay_airport.rename(columns={
		"airport":                   "DEST",
		"airport_arr_delay_rate":    "dest_arr_delay_rate",
		"airport_cancel_rate":       "dest_cancel_rate",
		"weather_delay_share":       "dest_weather_delay_share",
		"nas_delay_share":           "dest_nas_delay_share",
		"late_aircraft_delay_share": "dest_late_aircraft_delay_share",
		"carrier_delay_share":       "dest_carrier_delay_share",
	})
	df = (
		df
		.merge(origin_delay, on=["YEAR", "QUARTER", "ORIGIN"], how="left")
		.merge(dest_delay,   on=["YEAR", "QUARTER", "DEST"],   how="left")
	)

	# Core hypothesis variable: ratio of passengers carried to available seats.
	df["load_factor"] = _safe_divide(df["passengers_t100"], df["seats"])

	# Route-level delay features averaged across both endpoints.
	df["route_avg_arr_delay_rate"]      = df[["origin_arr_delay_rate",           "dest_arr_delay_rate"]].mean(axis=1)
	df["route_avg_cancel_rate"]         = df[["origin_cancel_rate",              "dest_cancel_rate"]].mean(axis=1)
	df["route_weather_delay_share"]     = df[["origin_weather_delay_share",      "dest_weather_delay_share"]].mean(axis=1)
	df["route_nas_delay_share"]         = df[["origin_nas_delay_share",          "dest_nas_delay_share"]].mean(axis=1)
	df["route_late_aircraft_delay_share"] = df[["origin_late_aircraft_delay_share", "dest_late_aircraft_delay_share"]].mean(axis=1)
	df["route_carrier_delay_share"]     = df[["origin_carrier_delay_share",      "dest_carrier_delay_share"]].mean(axis=1)

	# Routes with no competition record get zero — not missing.
	for col in ["competition_flights", "competition_unique_carriers", "competition_avg_delay",
	            "competition_cancel_rate", "competition_delay15_rate"]:
		if col in df.columns:
			df[col] = df[col].fillna(0)

	df["route"] = df["ORIGIN"] + "-" + df["DEST"]
	# Binary flag: routes at or above 80% load are considered capacity-saturated.
	df["is_saturated"] = (df["load_factor"] >= 0.8).astype(int)

	df = df.sort_values(["YEAR", "QUARTER", "ORIGIN", "DEST"]).reset_index(drop=True)

	if save:
		CLEANED_DIR.mkdir(exist_ok=True)
		df.to_csv(CLEANED_DIR / "analysis_table.csv", index=False)

	return df


if __name__ == "__main__":
	preprocess_all_data()
	build_analysis_table(save=True)
