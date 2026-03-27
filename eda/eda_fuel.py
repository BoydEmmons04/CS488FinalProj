# EDA for fuel prices
# daily gulf coast jet fuel jan-mar 2025

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensure_clean import ensure_cleaned_data

plot_dir = os.path.join(os.path.dirname(__file__), "plots", "fuel")
os.makedirs(plot_dir, exist_ok=True)

def main():
    cleaned_dir = ensure_cleaned_data()
    df = pd.read_csv(os.path.join(cleaned_dir, "fuel_cleaned.csv"), parse_dates=["observation_date"])
    df = df.rename(columns={"DJFUELUSGULF": "price"})

    #print(df.head())
    #print(df.tail())
    #print(df.sample(10))
    #print(df.info())
    #print(df.columns.tolist())
    #print(df.nunique())
    #print(df.duplicated().sum())

    print("\nFUEL: Descriptive Statistics")
    
    print("\nShape:", df.shape)
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nNumeric summary:")
    print(df["price"].describe())
    print("\nDate range:", df["observation_date"].min(), "to", df["observation_date"].max())
    print("Price range:", df["price"].min(), "to", df["price"].max())
    print("Price volatility (std):", round(df["price"].std(), 4))

    # time series of price 
    #Note: some null vals, so we dropna for the line chart but keep them for the histogram since the histogram can show the missing data as well
    ts = df.dropna(subset=["price"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts["observation_date"], ts["price"], color="black", linewidth=1.2)
    ax.fill_between(ts["observation_date"], ts["price"], alpha=0.1, color="gray")
    ax.set_title("Gulf Coast Jet Fuel Price, Q1 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($/gallon)")
    ax.axhline(df["price"].mean(), color="dimgray", linestyle="--",
               label="Mean=$" + str(round(df["price"].mean(), 3)))
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fuel_price_timeseries_linechart.png"))
    plt.close(fig)

    # histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["price"], bins=20, edgecolor="black", alpha=0.7, color="silver")
    ax.set_title("Fuel Price Distribution")
    ax.set_xlabel("Price ($/gallon)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fuel_price_distribution_histogram.png"))
    plt.close(fig)

    # daily changes in price barchart
    df["price_change"] = df["price"].diff()
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = ["green" if x >= 0 else "red" for x in df["price_change"].fillna(0)]
    ax.bar(df["observation_date"], df["price_change"].fillna(0), color=bar_colors, width=1)
    ax.set_title("Daily Fuel Price Change")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price Change ($/gallon)")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "daily_price_change_barchart.png"))
    plt.close(fig)

    #  7 day rolling avg
    df["rolling_7d"] = df["price"].rolling(window=7).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["observation_date"], df["price"], alpha=0.4, label="Daily", color="silver")
    ax.plot(df["observation_date"], df["rolling_7d"], color="black",
            linewidth=2, label="7-Day Rolling Avg")
    ax.set_title("Fuel Price with 7-Day Rolling Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($/gallon)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fuel_rolling_avg_linechart.png"))
    plt.close(fig)

    # weekly averages
    df["week"] = df["observation_date"].dt.isocalendar().week.astype(int)
    weekly = df.groupby("week")["price"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    weekly.plot.bar(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Average Weekly Fuel Price")
    ax.set_xlabel("Week Number")
    ax.set_ylabel("Avg Price ($/gallon)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "weekly_avg_price_barchart.png"))
    plt.close(fig)

    print("\nDaily price change stats:")
    print(df["price_change"].describe())
    print("\nDays with price increase:", (df["price_change"] > 0).sum())
    print("Days with price decrease:", (df["price_change"] < 0).sum())
    print("Days unchanged:", (df["price_change"] == 0).sum())
    print("\nPlots saved to", plot_dir)

if __name__ == "__main__":
    main()
