# Airline Load Factor and Airfare Analysis

I'm Carter Ward, a student at UAH, and this is my big-data analytics project on the U.S. airline industry. I built an end-to-end pipeline that pulls together five separate government and industry data sources — flight-level operations, ticket pricing, delay causes, jet fuel prices, and load factor/capacity data — cleans and merges them into a single route-level analysis table, and then uses that table to model what actually drives average airfare.

## 1. Purpose

This project exists to demonstrate that I can run a complete big-data pipeline on real, messy, multi-source data — not a toy dataset. I'm working with U.S. airline industry data covering competition, pricing, delays, fuel costs, and load factor (how full flights are) for Q1 2025, scoped to airports in California, Georgia, and Texas. The goal is a course project that shows the full lifecycle: collecting raw data at scale, cleaning and joining it, engineering features, applying multiple modeling techniques, and evaluating and interpreting the results honestly, including where the results don't confirm the starting hypothesis.

## 2. Problem and Approach

The core question I set out to test was whether higher airline **load factor** (the percentage of seats filled on a flight) is associated with higher **average airfare** — the intuition being that airlines with fuller planes have more pricing power. I treat `avg_fare` as the target variable and build a route-quarter feature set to predict it.

To answer this I didn't rely on a single technique. I ran three complementary methods on the same data, as laid out in `Model.py`:

1. **Correlation Analysis** — a focused Pearson correlation matrix across load factor, competition, delay, fuel, and fare variables, to see which relationships actually show up in the raw numbers before any modeling assumptions are applied.
2. **Linear Regression** — a straightforward, interpretable model over a fixed 19-feature set, trained on an 80/20 train/test split (`random_state=42` for reproducibility).
3. **PCA Regression** — the same features standardized and reduced with PCA (retaining 95% of variance) before regression, to check whether dimensionality reduction improves generalization or just throws away signal.

Running all three side by side let me cross-check the story: does the correlation matrix agree with what the regression coefficients say, and does compressing the feature space with PCA help or hurt prediction accuracy?

## 3. Structure and Methodologies

The codebase is split into three files, each owning one part of the pipeline, orchestrated by `Main.py`:

- **`Data.py`** — Stage 1 (preprocessing) and Stage 2 (feature engineering). It defines the target airports (11 in CA, 4 in GA, 11 in TX), reads the five raw sources, filters each to Q1 2025 and the target airports, writes cleaned CSVs to `Cleaned_Data/`, then aggregates and merges everything into a single route-quarter `Analysis_Table.csv`.
- **`Model.py`** — Stage 3 (feature selection), Stage 4 (the three modeling techniques), Stage 5 (evaluation metrics), and Stage 6 (visualization). It builds the model-ready frame, trains Linear Regression and PCA Regression, computes RMSE/MAPE/R²/SNR/accuracy, and generates every plot and text report in `outputs/`.
- **`Main.py`** — the entry point. It creates the output directories and runs the pipeline in order: preprocess → build analysis table → run models → evaluate and save outputs.

**Libraries used** (from `requirements.txt`): `pandas` and `numpy` for data wrangling and numerical work, `scikit-learn` for `LinearRegression`, `PCA`, `StandardScaler`, `train_test_split`, and the regression metrics, and `matplotlib` (in headless `Agg` mode) for every chart.

**Data structures/pipeline stages:**

1. Raw CSVs (per-flight, per-ticket, per-airport-month, per-day, or per-segment records).
2. Cleaned, filtered CSVs in `Cleaned_Data/` — one per source.
3. A route-quarter `Analysis_Table.csv` (`ORIGIN`, `DEST`, `YEAR`, `QUARTER` as the join keys) combining all five sources.
4. A model frame with lag/rolling load-factor features and a fixed 19-column feature set, median-imputed.
5. Trained model objects, predictions, and evaluation tables/plots.

## 4. Process

Here's how the build actually went, in the order the pipeline runs:

**Data collection.** I sourced five datasets covering different angles of the airline industry for Q1 2025:
- **Competition (Airline Count)** — a 93 MB flight-level dataset (`US Flights Data (2025, Q1) - Flight Dataset.csv`) with per-flight carrier, delay, and cancellation records.
- **DB1B Market Airline Ticket Data** — three state-level files (CA/GA/TX, ~104 MB combined) with per-market passenger counts, fares, and distances.
- **Flight Delays (Airline_Delay_Cause)** — monthly, per-airport delay-cause breakdowns (weather, NAS, carrier, late aircraft, security).
- **Fuel Prices (DJFUELUSGULF)** — daily U.S. Gulf Coast jet fuel price series.
- **T-100 (Load Factor)** — state-level segment data (CA/GA/TX) with departures, seats, and passengers per route.

**Cleaning.** Because the Competition and DB1B files are large, I read them in 100,000-row chunks (`pd.read_csv(..., chunksize=100_000)`) and filtered each chunk down to Q1 2025 and my 26 target airports before concatenating, rather than loading the full files into memory. Everything else was small enough to load directly. Each source got its numeric columns coerced and null-cleaned, then written out to `Cleaned_Data/` — the cleaned files total roughly 3.7 million rows (2.9M from DB1B alone, 744K from Competition).

**Feature engineering.** In `build_analysis_table`, I aggregated each cleaned source to the route-quarter grain (or airport-quarter for delays, quarter-only for fuel) and merged them on `ORIGIN`/`DEST`/`YEAR`/`QUARTER`. I computed `load_factor` as passengers divided by seats, built origin- and destination-side delay/cancellation shares and averaged them to route level, and flagged routes with `load_factor >= 0.8` as `is_saturated`. This produced `Analysis_Table.csv` with 2,850 route-quarter rows.

**Modeling prep.** In `_build_model_frame`, I added lag and 2-period rolling versions of `load_factor` per route, median-imputed any remaining gaps, and locked in a 19-feature set (load factor variants, distance, fuel price, passenger volume, competition metrics, and the six route-level delay/cancellation shares) with `avg_fare` as the target.

**Correlation, regression, and PCA.** I ran a Pearson correlation matrix, then trained Linear Regression directly on the 19 features and PCA Regression on the standardized, PCA-reduced version (95% variance retained, which took 10 components).

**Evaluation and visualization.** I computed RMSE, MAPE, R², and a signal-to-noise ratio for both models, then generated a correlation heatmap, scatter plots (load factor vs. fare, competition vs. fare, delay rate vs. fare), a five-panel model comparison chart, an actual-vs-predicted panel, and a six-panel diagnostics dashboard (fare/load-factor distributions, time trend, PCA cumulative variance).

## 5. Outcome

The pipeline produced concrete, citable results in `outputs/evaluation/` and `outputs/modeling/`:

- **Linear Regression won on every metric**: RMSE = $65.04, MAPE = 21.59%, R² = 0.388, accuracy = 78.41%, versus PCA Regression's RMSE = $68.14, MAPE = 23.35%, R² = 0.328, accuracy = 76.65% (`metrics.txt`). Compressing 19 features down to 10 principal components didn't help — it actually made predictions slightly worse, which was a useful lesson about not assuming dimensionality reduction is automatically better.
- **My original hypothesis wasn't well supported.** `load_factor`'s correlation with `avg_fare` is only 0.074 — essentially negligible (`correlation.txt`, `conclusions.txt`). The strongest correlate of fare turned out to be `market_distance` at 0.553, followed by `is_saturated` at 0.219. Distance drives price far more than how full the plane is.
- **The linear model's biggest coefficients are the delay-share features** — `route_weather_delay_share` (+4437.72), `route_avg_arr_delay_rate` (−2645.41), `route_nas_delay_share` (+1381.65) — much larger in magnitude than load factor's coefficient (−7.15) or `is_saturated`'s (+11.77). That gap between a near-zero raw correlation and a real (if small) coefficient for saturated routes tells me load factor has a conditional effect once distance and delay dynamics are held fixed, but it's not the dominant driver I expected going in.
- **PCA showed the feature set isn't very redundant**: it took 10 of 19 components to reach 95% cumulative variance (`pca_variance.txt`), with PC1 alone (load-factor dominated) explaining only 29.6%. The data doesn't compress cleanly, which is part of why PCA regression didn't outperform the plain linear model.
- **A real data limitation showed up in the correlation matrix**: `avg_fuel_price` correlates as `NaN` with everything (`correlation.txt`). Because the whole dataset is scoped to a single quarter (Q1 2025), the quarter-average fuel price is identical for every row after merging — zero variance, so no correlation can be computed. That's a direct consequence of the project's single-quarter scope, and it's the kind of thing you only catch by actually inspecting the output rather than trusting the pipeline blindly.
- On the held-out test set (570 rows), both models predicted a mean fare within about $6 of the actual mean ($269.43 actual vs. $275.26 predicted for Linear Regression), with a median absolute error around $32 and a 90th-percentile absolute error around $100 (`predictions_summary.txt`).

**What I took away from this.** Working across five differently-shaped government/industry datasets (flight-level, ticket-market-level, monthly-airport-level, daily, and segment-level) forced me to get comfortable with chunked reading for memory efficiency, choosing the right join keys and aggregation grain for each source, and being careful about what "clean" actually means for each file. More importantly, the project taught me to trust the evidence over the hypothesis — I went in expecting load factor to be the headline driver of fare, and the data said distance and delay dynamics mattered more. Reporting that honestly, instead of cherry-picking a result that confirmed my starting assumption, is the part of this project I'm most proud of.

## How to Run

```bash
pip install -r requirements.txt
python Main.py
```

This runs the full pipeline in order — preprocessing, feature engineering, modeling, and evaluation — and writes all cleaned data, the analysis table, and every output artifact listed above.
