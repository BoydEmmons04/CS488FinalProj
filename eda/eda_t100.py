# EDA for T-100 dataset
# this is data we need for load factor (passengers / seats)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensure_clean import ensure_cleaned_data

plot_dir = os.path.join(os.path.dirname(__file__), "plots", "t100")
os.makedirs(plot_dir, exist_ok=True)

def main():
    cleaned_dir = ensure_cleaned_data()
    df = pd.read_csv(os.path.join(cleaned_dir, "t100_cleaned.csv"))

    #print(df.head())
    #print(df.tail())
    #print(df.sample(10))
    #print(df.info())
    
    #print(df.columns.tolist())
    #print(df.nunique())
    
    #print(df.duplicated().sum())

    print("\nT-100: Descriptive Statistics")
    
    print("\nShape:", df.shape)
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nNumeric summary:")
    print(df.describe())
    print("\nUnique origins:", df["ORIGIN"].nunique())
    print("Unique destinations:", df["DEST"].nunique())

    # load factor,variable for project hypothesis
    df["LOAD_FACTOR"] = df["PASSENGERS"] / df["SEATS"].replace(0, np.nan)

    # filter to rows that actually had seats
    active = df[df["SEATS"] > 0].copy()
    print("\nRows with SEATS > 0:", len(active), "/", len(df))
    print("Rows with zero seats:", (df["SEATS"] == 0).sum())

    if len(active) == 0:
        print("WARNING: no active records found, using full data instead")
        active = df.copy()
        active["LOAD_FACTOR"] = 0

    # load factor distribution
    lf = active["LOAD_FACTOR"].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lf, bins=50, edgecolor="black", alpha=0.7, color="silver")
    ax.set_title("Load Factor Distribution")
    ax.set_xlabel("Load Factor (Passengers / Seats)")
    ax.set_ylabel("Frequency")
    if len(lf) > 0:
        ax.axvline(lf.mean(), color="black", linestyle="--",
                   label="Mean=" + str(round(lf.mean(), 2)))
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "load_factor_distribution_histogram.png"))
    plt.close(fig)

    print("\nLoad factor stats:")
    print(lf.describe())

    # passenger counts
    pax = active["PASSENGERS"]
    pax_clip = pax[pax < pax.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pax_clip, bins=50, edgecolor="black", alpha=0.7, color="darkgray")
    ax.set_title("Passenger Count Distribution (99th pctl clip)")
    ax.set_xlabel("Passengers")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "passengers_distribution_histogram.png"))
    plt.close(fig)

    # seats
    seats = active["SEATS"]
    seats_clip = seats[seats < seats.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(seats_clip, bins=50, edgecolor="black", alpha=0.7, color="silver")
    ax.set_title("Seat Count Distribution (99th pctl clip)")
    ax.set_xlabel("Seats")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "seats_distribution_histogram.png"))
    plt.close(fig)

    # top routes by passengers
    active["Route"] = active["ORIGIN"] + " to " + active["DEST"]
    route_pax = (active.groupby("Route")["PASSENGERS"]
        .sum().sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 6))
    route_pax.plot.barh(ax=ax, color="gray", edgecolor="black")
    ax.set_title("Top 15 Routes by Total Passengers")
    ax.set_xlabel("Total Passengers")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_routes_by_passengers_barchart.png"))
    plt.close(fig)

    # load factor for busiest routes
    route_lf = (active.groupby("Route")
        .agg(avg_lf=("LOAD_FACTOR","mean"), total_pax=("PASSENGERS","sum"))
        .sort_values("total_pax", ascending=False).head(15)
        .sort_values("avg_lf"))
    fig, ax = plt.subplots(figsize=(10, 6))
    route_lf["avg_lf"].plot.barh(ax=ax, color="darkgray", edgecolor="black")
    ax.set_title("Avg Load Factor for Top 15 Routes (by pax volume)")
    ax.set_xlabel("Average Load Factor")
    ax.axvline(0.8, color="black", linestyle="--", alpha=0.7, label="80% threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "avg_load_factor_top_routes_barchart.png"))
    plt.close(fig)

    # busiest departure airports
    origin_deps = (active.groupby("ORIGIN")["DEPARTURES_PERFORMED"]
        .sum().sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 6))
    origin_deps.plot.barh(ax=ax, color="silver", edgecolor="black")
    ax.set_title("Top 15 Origin Airports by Total Departures")
    ax.set_xlabel("Total Departures Performed")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top_origins_by_departures_barchart.png"))
    plt.close(fig)

    # passengers vs seats ,,diagonal = 100% load factor
    fig, ax = plt.subplots(figsize=(8, 7))
    sample = active.sample(min(5000, len(active)), random_state=42)
    ax.scatter(sample["SEATS"], sample["PASSENGERS"], alpha=0.5, s=18, color="dimgray", edgecolors="black", linewidths=0.3)
    max_val = max(sample["SEATS"].max(), sample["PASSENGERS"].max())
    ax.plot([0, max_val], [0, max_val], "k--", label="100% Load Factor")
    ax.set_title("Passengers vs. Seats")
    ax.set_xlabel("Seats")
    ax.set_ylabel("Passengers")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "passengers_vs_seats_scatterplot.png"))
    plt.close(fig)

    # correlation full heatmap 
    num_cols = ["DEPARTURES_PERFORMED","SEATS","PASSENGERS","LOAD_FACTOR"]
    corr = active[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, linecolor="white", linewidths=1,
                cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (T-100)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close(fig)

    # correlation heatmap lower triangle
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, annot=True, linecolor="white", linewidths=1,
                cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap – Lower Triangle (T-100)")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_heatmap_lower.png"))
    plt.close(fig)

    # pairplot of key T100 variables
    pair_cols = ["DEPARTURES_PERFORMED","SEATS","PASSENGERS","LOAD_FACTOR"]
    sample_pair = active[pair_cols].sample(min(3000, len(active)), random_state=42)
    g = sns.pairplot(sample_pair, plot_kws={"alpha": 0.4, "s": 12},
                     diag_kws={"edgecolor": "black"})
    g.figure.suptitle("Pairplot of T-100 Variables", y=1.02)
    g.savefig(os.path.join(plot_dir, "pairplot_t100.png"))
    plt.close(g.figure)

    # boxplots of key metrics 
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for idx, feat in enumerate(pair_cols):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(y=active[feat], ax=ax, color="steelblue", showfliers=False)
        ax.set_title(feat)
    plt.suptitle("Distribution of T-100 Variables", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "feature_boxplots.png"))
    plt.close(fig)

    print("\nCorrelation matrix:")
    print(corr)
    print("\nPlots saved to", plot_dir)

if __name__ == "__main__":
    main()
