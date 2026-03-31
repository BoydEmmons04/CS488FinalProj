import pandas as pd
from pathlib import Path

# Define data directory
DATA_DIR = Path("Project Datasets")

# Airport codes by state (major hubs)
CA_AIRPORTS = ["LAX", "SFO", "SAN", "OAK", "ONT", "BUR", "SJC", "SMF", "PSP", "FAT", "BFL"]
GA_AIRPORTS = ["ATL", "SAV", "AGS", "ABY"]
TX_AIRPORTS = ["DFW", "IAH", "DAL", "HOU", "AUS", "SAT", "ELP", "MAF", "ABI", "AMA", "GRK"]
TARGET_AIRPORTS = set(CA_AIRPORTS + GA_AIRPORTS + TX_AIRPORTS)


def load_competition_data():
	"""Load and clean US Flights Data (2025 Q1)."""
	file_path = DATA_DIR / "Competition (Airline Count)" / "US Flights Data (2025, Q1) - Flight Dataset.csv"

	# Use chunks for the larger competition file.
	chunks = []
	for chunk in pd.read_csv(file_path, chunksize=100000):
		chunk["Date"] = pd.to_datetime(chunk["Date"])
		chunk = chunk[(chunk["Date"] >= "2025-01-01") & (chunk["Date"] <= "2025-03-31")]
		chunk = chunk[chunk["Origin"].isin(TARGET_AIRPORTS) | chunk["Dest"].isin(TARGET_AIRPORTS)]
		if not chunk.empty:
			chunks.append(chunk)

	if chunks:
		df = pd.concat(chunks, ignore_index=True)
	else:
		df = pd.DataFrame()

	if not df.empty:
		df["Dep_Time"] = df["Dep_Time"].astype(str).str.zfill(4)
		df["Actual_Dep"] = df["Actual_Dep"].astype(str).str.zfill(4)
		df["Delay"] = pd.to_numeric(df["Delay"], errors="coerce")
		df["Cancelled"] = df["Cancelled"].astype(int)

	return df


def load_db1b_data(state):
	"""Load DB1B Market data for a given state."""
	file_path = DATA_DIR / "DB1BMarket Airline Ticket Data" / f"{state}_T_DB1B_MARKET.csv"
	df = pd.read_csv(file_path)

	df = df[(df["YEAR"] == 2025) & (df["QUARTER"] == 1)]
	df["PASSENGERS"] = pd.to_numeric(df["PASSENGERS"], errors="coerce")
	df["MARKET_FARE"] = pd.to_numeric(df["MARKET_FARE"], errors="coerce")
	df["MARKET_DISTANCE"] = pd.to_numeric(df["MARKET_DISTANCE"], errors="coerce")

	return df


def load_delay_data():
	"""Load and clean Airline Delay Cause data."""
	file_path = DATA_DIR / "Flight Delays" / "ot_delaycause1_DL" / "Airline_Delay_Cause.csv"
	df = pd.read_csv(file_path)

	df = df[(df["year"] == 2025) & (df["month"].isin([1, 2, 3]))]
	df = df[df["airport"].isin(TARGET_AIRPORTS)]

	numeric_cols = [
		"arr_flights",
		"arr_del15",
		"carrier_ct",
		"weather_ct",
		"nas_ct",
		"security_ct",
		"late_aircraft_ct",
		"arr_cancelled",
		"arr_diverted",
		"arr_delay",
		"carrier_delay",
		"weather_delay",
		"nas_delay",
		"security_delay",
		"late_aircraft_delay",
	]
	for col in numeric_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	return df


def load_fuel_data():
	"""Load and clean fuel prices data."""
	file_path = DATA_DIR / "Fuel Prices" / "DJFUELUSGULF.csv"
	df = pd.read_csv(file_path)

	df["observation_date"] = pd.to_datetime(df["observation_date"])
	df = df[(df["observation_date"] >= "2025-01-01") & (df["observation_date"] <= "2025-03-31")]
	df["DJFUELUSGULF"] = pd.to_numeric(df["DJFUELUSGULF"], errors="coerce")

	return df


def load_t100_data(state):
	"""Load T-100 Load Factor data for a given state."""
	file_path = DATA_DIR / "T-100 (Load Factor)" / f"{state}_T_T100D_SEGMENT_US_CARRIER_ONLY.csv"
	df = pd.read_csv(file_path)

	df = df[(df["YEAR"] == 2025) & (df["QUARTER"] == 1)]
	df["DEPARTURES_PERFORMED"] = pd.to_numeric(df["DEPARTURES_PERFORMED"], errors="coerce")
	df["SEATS"] = pd.to_numeric(df["SEATS"], errors="coerce")
	df["PASSENGERS"] = pd.to_numeric(df["PASSENGERS"], errors="coerce")

	return df


def preprocess_all_data(output_dir=Path("cleaned_data")):
	"""Load, filter, and persist all cleaned datasets."""
	print("Loading DB1B data...")
	db1b_df = pd.concat(
		[load_db1b_data("CA"), load_db1b_data("GA"), load_db1b_data("TX")],
		ignore_index=True,
	)

	print("Loading fuel data...")
	fuel_df = load_fuel_data()

	print("Loading T-100 data...")
	t100_df = pd.concat(
		[load_t100_data("CA"), load_t100_data("GA"), load_t100_data("TX")],
		ignore_index=True,
	)

	output_dir.mkdir(exist_ok=True)
	db1b_df.to_csv(output_dir / "db1b_cleaned.csv", index=False)
	fuel_df.to_csv(output_dir / "fuel_cleaned.csv", index=False)
	t100_df.to_csv(output_dir / "t100_cleaned.csv", index=False)

	print("Preprocessing complete. Cleaned datasets saved to 'cleaned_data' folder.")

	return {
		"db1b": db1b_df,
		"fuel": fuel_df,
		"t100": t100_df,
	}


if __name__ == "__main__":
	preprocess_all_data()