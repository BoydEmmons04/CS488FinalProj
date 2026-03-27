# EDA for db1b (airfare) dataset
# ticket level fare data for CA, GA, TX destinations

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensure_clean import ensure_cleaned_data

plot_dir = os.path.join(os.path.dirname(__file__), "plots", "db1b")
os.makedirs(plot_dir, exist_ok=True)


def main():
    cleaned_dir = ensure_cleaned_data()
    df = pd.read_csv(os.path.join(cleaned_dir, "db1b_cleaned.csv"))

    #print(df.head())
    #print(df.tail())
    #print(df.sample(10))
    #print(df.info())
    #print(df.columns.tolist())
    #print(df.nunique())
    #print(df.duplicated().sum())

    print("\nDB1B: Descriptive Statistics")
    
    print("\nShape:", df.shape)
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nNumeric summary:")
    print(df.describe())
    print("\nUnique origins:", df["ORIGIN"].nunique())
    print("Unique destinations:", df["DEST"].nunique())
    print("Fare range:", df["MARKET_FARE"].min(), "to", df["MARKET_FARE"].max())
    print("Mean fare:", round(df["MARKET_FARE"].mean(), 2))
    print("Median fare:", round(df["MARKET_FARE"].median(), 2))

    #fair histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["MARKET_FARE"], bins=60, edgecolor="black", alpha=0.7, color="silver")
    ax.set_title("Distribution of Market Fares")
    ax.set_xlabel("Fare (USD)")
    ax.set_ylabel("Frequency")
    ax.ticklabel_format(axis="y", style="plain")
    ax.axvline(df["MARKET_FARE"].mean(), color="black", linestyle="--",
               label="Mean=$" + str(round(df["MARKET_FARE"].mean())))
    ax.axvline(df["MARKET_FARE"].median(), color="dimgray", linestyle=":",
               label="Median=$" + str(round(df["MARKET_FARE"].median())))
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fare_distribution_histogram.png"))
    plt.close(fig)

    #distance distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["MARKET_DISTANCE"], bins=50, edgecolor="black", alpha=0.7, color="darkgray")
    ax.set_title("Distribution of Market Distance")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Frequency")
    ax.axvline(df["MARKET_DISTANCE"].mean(), color="black", linestyle="--",
               label="Mean=" + str(round(df["MARKET_DISTANCE"].mean())) + " mi")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "distance_distribution_histogram.png"))
    plt.close(fig)

    # scatter, does fare go up with distance?
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["MARKET_DISTANCE"], sample["MARKET_FARE"], alpha=0.5, s=18, color="dimgray", edgecolors="black", linewidths=0.3)
    ax.set_title("Market Fare vs. Distance (sampled)")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Fare (USD)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fare_vs_distance_scatterplot.png"))
    plt.close(fig)

    # top destinations
    top_dest = df["DEST"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_dest.plot.barh(ax=ax, color="gray", edgecolor="black")
    ax.set_title("Top 15 Destinations by Number of Tickets")
    ax.set_xlabel("Number of Tickets")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_destinations_barchart.png"))
    plt.close(fig)

    # avg fare at those top destinations
    top_dest_codes = df["DEST"].value_counts().head(15).index
    avg_fare_dest = (df[df["DEST"].isin(top_dest_codes)]
                     .groupby("DEST")["MARKET_FARE"].mean().sort_values())
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_fare_dest.plot.barh(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Average Fare by Destination (Top 15 by Volume)")
    ax.set_xlabel("Average Fare (USD)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "avg_fare_by_destination_barchart.png"))
    plt.close(fig)

    # top origins
    top_origins = df["ORIGIN"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_origins.plot.barh(ax=ax, color="silver", edgecolor="black")
    ax.set_title("Top 15 Origin Airports by Ticket Count")
    ax.set_xlabel("Number of Tickets")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_origins_barchart.png"))
    plt.close(fig)

    # fare per mile
    df["FARE_PER_MILE"] = df["MARKET_FARE"] / df["MARKET_DISTANCE"].replace(0, np.nan)
    fpm = df["FARE_PER_MILE"].dropna()
    fpm_clipped = fpm[fpm < fpm.quantile(0.99)]  # clip top 1% outliers

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(fpm_clipped, bins=60, edgecolor="black", alpha=0.7, color="gray")
    ax.set_title("Fare per Mile Distribution (99th percentile clip)")
    ax.set_xlabel("Fare per Mile (USD/mi)")
    ax.set_ylabel("Frequency")
    ax.axvline(fpm_clipped.mean(), color="black", linestyle="--",
               label="Mean=$" + str(round(fpm_clipped.mean(), 3)) + "/mi")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "fare_per_mile_histogram.png"))
    plt.close(fig)

    # correlation full heatmap 
    num_cols = ["PASSENGERS", "MARKET_FARE", "MARKET_DISTANCE"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, linecolor="white", linewidths=1,
                cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (DB1B)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close(fig)

    # correlation heatmap lower tri
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, annot=True, linecolor="white", linewidths=1,
                cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap – Lower Triangle (DB1B)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_heatmap_lower.png"))
    plt.close(fig)

    # pairplot colored by destination state
    
    state_map = {}
    tx_airports = ["AUS","DFW","DAL","HOU","IAH","SAT","ELP","LBB","MAF","AMA","CRP","HRL","GRK","ACT","SPS","TYR","GGG","BPT","ABI","SJT","CLL","LRD","MFE","BRO"]
    ga_airports = ["ATL","SAV","AGS","CSG","ABY","VLD","MCN","BQK"]
    ca_airports = ["LAX","SFO","SAN","SJC","OAK","SMF","ONT","BUR","SNA","LGB","FAT","PSP","SBP","MRY","RDD","ACV","STS","IYK","CEC","MOD","SCK","BFL"]
    for a in tx_airports: state_map[a] = "TX"
    for a in ga_airports: state_map[a] = "GA"
    for a in ca_airports: state_map[a] = "CA"
    df["DEST_STATE"] = df["DEST"].map(state_map).fillna("Other")

    pair_cols = ["PASSENGERS", "MARKET_FARE", "MARKET_DISTANCE", "DEST_STATE"]
    sample_pair = df[pair_cols].sample(min(3000, len(df)), random_state=42)
    g = sns.pairplot(sample_pair, hue="DEST_STATE", plot_kws={"alpha": 0.5, "s": 15})
    g.figure.suptitle("Pairplot by Destination State (DB1B)", y=1.02)
    g.savefig(os.path.join(plot_dir, "pairplot_by_dest_state.png"))
    plt.close(g.figure)

    # boxplots of fare by destination state 
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    features = ["MARKET_FARE", "MARKET_DISTANCE", "PASSENGERS", "FARE_PER_MILE"]
    titles = ["Market Fare", "Market Distance", "Passengers", "Fare per Mile"]
    box_df = df[df["DEST_STATE"] != "Other"].copy()
    for idx, (feat, title) in enumerate(zip(features, titles)):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(x="DEST_STATE", y=feat, hue="DEST_STATE", data=box_df, ax=ax,
                    palette="Set2", showfliers=False, legend=False)
        ax.set_title(title + " by Destination State")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "feature_boxplots_by_state.png"))
    plt.close(fig)

    print("\nCorrelation matrix:")
    print(corr)
    print("\nPlots saved to", plot_dir)

if __name__ == "__main__":
    main()
