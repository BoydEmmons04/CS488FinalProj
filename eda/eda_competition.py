# EDA for the competition dataset 
# US flights Q1 2025

import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from ensure_clean import ensure_cleaned_data

plot_dir = os.path.join(os.path.dirname(__file__), "plots", "competition")
os.makedirs(plot_dir, exist_ok=True)

def main():
    cleaned_dir = ensure_cleaned_data()
    df = pd.read_csv(os.path.join(cleaned_dir, "competition_cleaned.csv"), parse_dates=["Date"])

    #print(df.head())
    #print(df.tail())
    
    #print(df.sample(10))
    #print(df.info())
    
    #print(df.columns.tolist())
    #print(df.nunique())
    #print(df.duplicated().sum())

    # descriptive stats

    print("\nCompeytition: Descriptive Statistics")
    
    print("\nShape:", df.shape)
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nNumeric summary:")
    print(df.describe())
    print("\nUnique carriers:", df["Carrier"].nunique())
    print("Unique origins:", df["Origin"].nunique())
    print("Unique destinations:", df["Dest"].nunique())
    print("Date range:", df["Date"].min(), "to", df["Date"].max())
    print("Cancellation rate:", round(df["Cancelled"].mean() * 100, 2), "%")

    # delay distribution for non cancellled flights
   
    non_cancelled = df[df["Cancelled"] == 0]["Delay"].dropna()
    clipped = non_cancelled[(non_cancelled >= -60) & (non_cancelled <= 300)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clipped, bins=80, edgecolor="black", alpha=0.7, color="silver")
    ax.set_title("Distribution of Flight Delays (minutes)")
    ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Frequency")
    ax.axvline(0, color="black", linestyle="--", label="On-time")
    ax.axvline(non_cancelled.mean(), color="dimgray", linestyle="--",
               label="Mean=" + str(round(non_cancelled.mean(), 1)) + " min")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_distribution_histogram.png"))
    plt.close(fig)

    print("\nDelay stats (non-cancelled):")
    print(non_cancelled.describe())

    # top airlines by flight count
    top_carriers = df["Airline Name"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_carriers.plot.barh(ax=ax, color="gray", edgecolor="black")
    ax.set_title("Top 15 Airlines by Number of Flights")
    ax.set_xlabel("Number of Flights")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_carriers_flights_barchart.png"))
    plt.close(fig)

    # avg delay for top carriers
    top_carrier_codes = df["Carrier"].value_counts().head(15).index
    carrier_delay = (df[df["Carrier"].isin(top_carrier_codes)]
                     .groupby("Airline Name")["Delay"].mean().sort_values())
    fig, ax = plt.subplots(figsize=(10, 6))
    carrier_delay.plot.barh(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Average Delay by Airline (Top 15 by Volume)")
    ax.set_xlabel("Average Delay (minutes)")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "avg_delay_by_carrier_barchart.png"))
    plt.close(fig)

    # cancellation rate
    cancel_rate = (df[df["Carrier"].isin(top_carrier_codes)]
                   .groupby("Airline Name")["Cancelled"].mean().sort_values() * 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    cancel_rate.plot.barh(ax=ax, color="silver", edgecolor="black")
    ax.set_title("Cancellation Rate by Airline (Top 15 by Volume)")
    ax.set_xlabel("Cancellation Rate (%)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "cancellation_rate_by_carrier_barchart.png"))
    plt.close(fig)

    # flights per day over Q1
    daily_flights = df.groupby("Date").size()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_flights.index, daily_flights.values, color="black", linewidth=0.8)
    ax.set_title("Daily Flight Volume, Q1 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Flights")
    ax.axhline(daily_flights.mean(), color="gray", linestyle="--",
               label="Mean=" + str(round(daily_flights.mean())))
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "daily_flight_volume_linechart.png"))
    plt.close(fig)

    # bussiest routes
    
    df["Route"] = df["Origin"] + " to " + df["Dest"]
    top_routes = df["Route"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_routes.plot.barh(ax=ax, color="gray", edgecolor="black")
    ax.set_title("Top 15 Busiest Routes")
    ax.set_xlabel("Number of Flights")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_routes_barchart.png"))
    plt.close(fig)

    # check if any day of week has noticeably worse delays
    df["DayOfWeek"] = df["Date"].dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_delay = df.groupby("DayOfWeek")["Delay"].mean().reindex(day_order)
    fig, ax = plt.subplots(figsize=(8, 5))
    dow_delay.plot.bar(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Average Delay by Day of Week")
    ax.set_ylabel("Average Delay (minutes)")
    ax.set_xticklabels(day_order, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_by_day_of_week_barchart.png"))
    plt.close(fig)

    # are morning flights more delayed than evening ones
    df["DepHour"] = df["Dep_Time"].astype(int) // 100
    hourly_delay = df.groupby("DepHour")["Delay"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_delay.plot.bar(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Average Delay by Departure Hour")
    ax.set_xlabel("Departure Hour (0-23)")
    ax.set_ylabel("Average Delay (minutes)")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_by_departure_hour_barchart.png"))
    plt.close(fig)

    # boxplot of delay distributions by top 8 carriers
    top8 = df["Carrier"].value_counts().head(8).index
    box_carrier = df[(df["Carrier"].isin(top8)) & (df["Cancelled"] == 0)].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Airline Name", y="Delay", hue="Airline Name", data=box_carrier, ax=ax,
                palette="Set2", showfliers=False, legend=False)
    ax.set_title("Delay Distribution by Airline (Top 8 by Volume)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_boxplot_by_carrier.png"))
    plt.close(fig)

    # boxplot /violin of delay by day of week 
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_df = df[df["Cancelled"] == 0].copy()
    dow_df["DayOfWeek"] = dow_df["Date"].dt.day_name()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(x="DayOfWeek", y="Delay", hue="DayOfWeek", data=dow_df, order=day_order,
                ax=axes[0], palette="Set2", showfliers=False, legend=False)
    axes[0].set_title("Delay by Day of Week (Box)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_xlabel("")
    sns.violinplot(x="DayOfWeek", y="Delay", hue="DayOfWeek", data=dow_df, order=day_order,
                   ax=axes[1], palette="Set2", inner="quartile", cut=0, legend=False)
    axes[1].set_title("Delay by Day of Week (Violin)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(-60, 200)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_xlabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_boxplot_by_day_of_week.png"))
    plt.close(fig)

    # busiest origin airports
    top_origins = df["Origin"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_origins.plot.barh(ax=ax, color="silver", edgecolor="black")
    ax.set_title("Top 15 Busiest Origin Airports")
    ax.set_xlabel("Number of Departures")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_origin_airports_barchart.png"))
    plt.close(fig)

    print("\nPlots saved to", plot_dir)

if __name__ == "__main__":
    main()
