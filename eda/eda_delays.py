# EDA for the delays dataset
# has monthly delay stats by carrier and airport (TX, GA, CA)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensure_clean import ensure_cleaned_data

plot_dir = os.path.join(os.path.dirname(__file__), "plots", "delays")
os.makedirs(plot_dir, exist_ok=True)

def main():
    cleaned_dir = ensure_cleaned_data()
    df = pd.read_csv(os.path.join(cleaned_dir, "delays_cleaned.csv"))

    #print(df.head())
    #print(df.tail())
    #print(df.sample(10))
    #print(df.info())
    #print(df.columns.tolist())
    #print(df.nunique())
    #print(df.duplicated().sum())

    print("\nDelays: Descriptive Statistics")
    
    print("\nShape:", df.shape)

    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNumeric summary:")
    print(df.describe())
    print("\nUnique carriers:", df["carrier"].nunique())
    print("Unique airports:", df["airport"].nunique())
    print("Months covered:", sorted(df["month"].unique()))

    df["delay_rate"] = df["arr_del15"] / df["arr_flights"].replace(0, np.nan) * 100

    # look at what causes the delays
    
    cause_cols = ["carrier_delay","weather_delay","nas_delay",
                  "security_delay","late_aircraft_delay"]
    cause_totals = df[cause_cols].sum()
    labels = ["Carrier","Weather","NAS","Security","Late Aircraft"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pie_grays = ["#4a4a4a","#7a7a7a","#a0a0a0","#c0c0c0","#d9d9d9"]
    axes[0].pie(cause_totals, labels=labels, autopct="%1.1f%%",
                colors=pie_grays, startangle=140)
    axes[0].set_title("Delay Causes (Total Minutes)")

    cause_totals.index = labels
    cause_totals.plot.bar(ax=axes[1], color=pie_grays, edgecolor="black")
    axes[1].set_title("Total Delay Minutes by Cause")
    axes[1].set_ylabel("Total Delay (minutes)")
    axes[1].set_xticklabels(labels, rotation=45, ha="right")


    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_causes_breakdown_piechart.png"))
    plt.close(fig)
    print("\nDelay cause totals (minutes):")
    print(cause_totals)

    # delay rate by carrier
    carrier_stats = (df.groupby("carrier_name").agg(
        total_flights=("arr_flights","sum"),
        total_delayed=("arr_del15","sum"))
        .query("total_flights > 0"))

    carrier_stats["delay_rate"] = carrier_stats["total_delayed"] / carrier_stats["total_flights"] * 100
    carrier_stats = carrier_stats.sort_values("delay_rate")

    fig, ax = plt.subplots(figsize=(10, 8))
    carrier_stats["delay_rate"].plot.barh(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Delay Rate by Carrier (% Flights Delayed >15 min)")
    ax.set_xlabel("Delay Rate (%)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_rate_by_carrier_barchart.png"))
    plt.close(fig)

    # which airports have the most total delay
    airport_delays = (df.groupby("airport_name")["arr_delay"]
        .sum().sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 6))
    airport_delays.plot.barh(ax=ax, color="gray", edgecolor="black")
    ax.set_title("Top 15 Airports by Total Arrival Delay (minutes)")
    ax.set_xlabel("Total Arrival Delay (minutes)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_airports_by_delay_barchart.png"))
    plt.close(fig)

    # cancellations
    cancel_by_carrier = (df.groupby("carrier_name")["arr_cancelled"]
        .sum().sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 6))
    cancel_by_carrier.plot.barh(ax=ax, color="silver", edgecolor="black")
    ax.set_title("Total Cancellations by Carrier (Top 15)")
    ax.set_xlabel("Number of Cancellations")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "cancellations_by_carrier_barchart.png"))
    plt.close(fig)

    # monthly delay rate
    monthly = (df.groupby("month")
        .agg(total_flights=("arr_flights","sum"),
             total_delayed=("arr_del15","sum"))
        .query("total_flights > 0"))
    monthly["delay_rate"] = monthly["total_delayed"] / monthly["total_flights"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    monthly["delay_rate"].plot.bar(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Delay Rate by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Delay Rate (%)")
    ax.set_xticklabels([f"Month {int(m)}" for m in monthly.index], rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_rate_by_month_barchart.png"))
    plt.close(fig)

    # correlation fullheatmap 
    delay_cols = ["arr_flights","arr_del15","arr_delay",
                  "carrier_delay","weather_delay","nas_delay",
                  "security_delay","late_aircraft_delay"]
    corr = df[delay_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, linecolor="white", linewidths=1,
                cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f",
                annot_kws={"fontsize": 7}, ax=ax)
    ax.set_title("Correlation Heatmap (Delay Variables)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close(fig)

    # correlation heatmap lower triangle
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, annot=True, linecolor="white", linewidths=1,
                cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f",
                annot_kws={"fontsize": 7}, ax=ax)
    ax.set_title("Correlation Heatmap – Lower Triangle (Delay Variables)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_heatmap_lower.png"))
    plt.close(fig)

    # pairplot of key delay variables
    pair_cols = ["arr_flights", "arr_del15", "arr_delay",
                 "carrier_delay", "weather_delay", "nas_delay",
                 "late_aircraft_delay"]
    pair_df = df[pair_cols].dropna()
    sample_pair = pair_df.sample(min(3000, len(pair_df)), random_state=42)
    g = sns.pairplot(sample_pair, plot_kws={"alpha": 0.4, "s": 12},
                     diag_kws={"edgecolor": "black"})
    g.figure.suptitle("Pairplot of Delay Variables", y=1.02)
    g.savefig(os.path.join(plot_dir, "pairplot_delays.png"))
    plt.close(g.figure)

    # boxplots of delay types by top 6 carriers 
    top6 = (df.groupby("carrier_name")["arr_flights"].sum()
            .sort_values(ascending=False).head(6).index)
    box_df = df[df["carrier_name"].isin(top6)].copy()
    cause_cols_plot = ["carrier_delay","weather_delay","nas_delay",
                       "security_delay","late_aircraft_delay","arr_delay"]
    cause_titles = ["Carrier Delay","Weather Delay","NAS Delay",
                    "Security Delay","Late Aircraft Delay","Total Arr Delay"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (col, title) in enumerate(zip(cause_cols_plot, cause_titles)):
        ax = axes[idx // 3, idx % 3]
        sns.boxplot(x="carrier_name", y=col, hue="carrier_name", data=box_df, ax=ax,
                    palette="Set2", showfliers=False, legend=False)
        ax.set_title(title + " by Carrier")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    plt.suptitle("Delay Type Distributions by Carrier (Top 6)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "delay_boxplots_by_carrier.png"))
    plt.close(fig)

    # diversions
    divert = (df.groupby("airport_name")["arr_diverted"]
        .sum().sort_values(ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(10, 5))
    divert.plot.barh(ax=ax, color="gray", edgecolor="black")
    ax.set_title("Top 10 Airports by Diversions")
    ax.set_xlabel("Number of Diversions")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "diversions_by_airport_barchart.png"))
    plt.close(fig)

    print("\nPlots saved to", plot_dir)

if __name__ == "__main__":
    main()
