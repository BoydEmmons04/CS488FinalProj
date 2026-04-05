# Airline Load Factor and Airfare Analysis

**CS 488 — Final Project**

Carter Boyd — A#

---

## Table of Contents

1. [Introduction](#introduction)
2. [Discussion of Methods](#discussion-of-methods)
3. [Results, Data Visualization and Analysis](#results-data-visualization-and-analysis)
4. [Conclusion](#conclusion)
5. [References](#references)
6. [Appendix: Additional Results and Code](#appendix-additional-results-and-code)

---

## Introduction

Airfare pricing is shaped by a combination of operational, economic, and market factors, making it a useful topic for both industry analysis and consumer welfare research. In the U.S. airline industry, capacity management, route-level competition, fuel costs, and operational disruptions such as delays are commonly cited determinants of fares.

A central indicator of how airlines balance capacity and demand is the load factor, defined as the share of available seats that are sold on a given flight. In recent years, load factors have remained high across much of the industry, raising an important empirical question. When flights operate near capacity on a sustained basis, do airlines obtain greater pricing power, or do competitive and cost constraints limit fare increases? This question motivates the project's primary hypothesis:

**Sustained increases in airline load factors above historical norms lead to disproportionate increases in average airfares, even when fuel prices and total passenger demand are held constant.**

The initial research design aimed to test this relationship across multiple years and a broader geographic scope. During preliminary data review, however, it became evident that integrating DB1B market data (the Bureau of Transportation Statistics' Origin and Destination Survey, which records itinerary-level ticket information including fares, origin–destination pairs, and passenger counts), T-100 segment data (the BTS Air Carrier Statistics database, which reports carrier-level capacity, traffic, and load factor by route segment), delay records, competition measures, and fuel price series across several years would likely introduce inconsistencies in coverage, variable definitions, and reporting practices. Such inconsistencies can weaken internal validity and reduce confidence that observed effects reflect true relationships rather than measurement differences.

To strengthen comparability across sources, the analysis is limited to the first quarter of 2025, defined as January through March, and to routes associated with airports in Texas, Georgia, and California. These states were selected because they contain large and heterogeneous air travel markets, including a mixture of hub and non-hub airports and a range of route structures. Restricting the study to a single quarter improves temporal alignment across datasets and reduces the risk of confounding introduced by structural changes over longer horizons.

The empirical approach combines DB1B itinerary-level fare information with T-100 measures of capacity and load factor, as well as route-level delay indicators, competition metrics, and fuel prices. This integrated structure allows airfare outcomes to be evaluated in relation to load factor while accounting for other factors that plausibly influence pricing. Narrowing the scope in this manner provides a consistent observational window and creates a clearer basis for assessing whether load factor is systematically related to airfares after key controls are introduced.

## Discussion of Methods

Given the integrated dataset described above, the analysis requires methods that can (1) measure the raw association between load factor and fare, (2) estimate load factor's effect while holding other variables constant, and (3) assess whether correlated features inflate or obscure that effect. Three techniques were selected to meet these requirements.

The analysis table is split 80/20 into training and test sets using a fixed random seed for reproducibility. All models are trained on the same split to ensure results are comparable across techniques.

The first technique we used was Correlation Analysis. Pearson correlation serves as an initial diagnostic. Because the hypothesis implies a positive directional relationship between load factor and fare, computing pairwise correlations between average fare and all numeric features in the analysis table helps confirm whether the expected association appears in the data and where load factor ranks relative to alternative predictors such as distance, competition, and delay measures.

The second technique we used was Linear Regression. Linear Regression is the primary model because it produces a signed coefficient for each of the 19 features. This directly answers the central question: holding the remaining 18 variables constant, how much does a one-unit change in load factor change fare? This interpretability is essential for hypothesis testing, where the direction and magnitude of load factor's effect (not just overall prediction accuracy) are the quantities of interest. No regularization is applied, since the objective is to preserve coefficient magnitudes for comparison rather than to optimize out-of-sample performance.

The third technique we used was PCA Regression. PCA followed by Linear Regression is included to evaluate whether redundancy among the 19 features affects the linear model's coefficient estimates. Several features are correlated by construction (for example, the four load factor variants and the multiple delay-share measures), and PCA compresses these into orthogonal components ordered by variance explained. Comparing PCA Regression's performance against the unreduced Linear Regression helps indicate whether the main dimensions of variation also carry the fare signal, or whether compression removes information relevant to fare prediction.

The methodology has known limitations. The single-quarter design precludes testing the "sustained over time" dimension of the hypothesis, since lagged and rolling load factor features collapse to the current value when no prior quarters exist. Route-quarter aggregation also smooths route-specific pricing variation, and linear models assume a constant marginal effect that may not capture threshold or nonlinear dynamics.

## Results, Data Visualization and Analysis

The results are organized around three components of the hypothesis: (1) whether load factor is correlated with fare, (2) whether that relationship is disproportionate, and (3) whether it persists after controlling for fuel prices and demand. Full model diagnostics, complete feature rankings, PCA variance tables, and distance-based comparisons are provided in the Appendix.

### Load factor exhibits a weak correlation with fare

Across all 30 numeric features in the dataset, load factor ranks 16th in its Pearson correlation with average fare, at r = 0.0725, accounting for approximately 0.5% of fare variance (r² = 0.005). By comparison, market distance — the strongest correlate — has r = 0.5532 (approximately 30.6% of variance), and the binary `is_saturated` flag (≥80% load factor) has r = 0.2191. The full correlation ranking is provided in Appendix A.6 (source: full correlation matrix computed in `Model.py`, key values summarized in `Outputs/Evaluation/Conclusions.csv`).

The scatter plot of load factor versus airfare (source: `Outputs/Modeling/Modeling_Overview.png`, subplot "Load Factor vs Airfare") shows widely dispersed points with no visible upward trend, consistent with this weak correlation.

### The fare–load factor relationship is not disproportionate

The hypothesis predicts that fares rise disproportionately as load factor increases. A load factor quartile analysis (derived from `Outputs/Analysis_Table.csv`) provides a direct test:

| Load Factor Quartile | Mean Fare | Median Fare | Routes |
|---|---|---|---|
| Q1 (lowest load factors) | $256.66 | $252.06 | 682 |
| Q2 | $251.20 | $243.74 | 681 |
| Q3 | $279.44 | $271.73 | 681 |
| Q4 (highest load factors) | $305.15 | $298.22 | 681 |

The pattern does not indicate a disproportionate increase. The second quartile has the lowest mean fare ($251.20), below even the first ($256.66). The only notable increase occurs between Q3 and Q4, a $25.71 difference that coincides with routes where the `is_saturated` flag is active. Saturated routes (n = 936) have a mean fare of $301.06 versus $261.54 for non-saturated routes (n = 1,913), a raw difference of $39.52 (15.1%). After controlling for all 18 other features in the linear model, the `is_saturated` coefficient is reduced to +$11.77, and the load factor coefficient itself is −$7.15 — opposite in direction to what the hypothesis predicts.

This suggests that capacity pressure, to the extent it influences fare, operates as a threshold effect at high load levels rather than as the continuous disproportionate increase described by the hypothesis.

### Fuel price and demand controls lack sufficient variation

The hypothesis specifies that the load factor–fare relationship should hold even when fuel prices and demand are accounted for. In this dataset:

- **Fuel price** is constant at $2.2231 across all 2,849 rows (std = 0.0000), producing undefined (NaN) correlations and a regression coefficient of exactly $0.00. With zero variance, fuel price cannot function as a control variable.
- **Demand** (`passengers_db1b`) has a Pearson correlation of 0.0023 with fare, effectively zero.

Because neither variable varies meaningfully within a single-quarter cross-section, the hypothesis condition "even when fuel prices and demand are held constant" cannot be evaluated in this design. The focused correlation matrix illustrating these relationships is in Appendix A.7 (source: `Outputs/Modeling/Correlation.csv`).

### Summary of results

Load factor's raw correlation with fare is 0.073 (rank 16 of 30), its controlled coefficient is −$7.15, and the quartile analysis shows no progressive fare increase with rising load factor. The only directionally consistent signal is the saturation threshold (+$11.77 after controls), which represents a modest discrete effect. The hypothesis's required controls — fuel price and demand — do not vary sufficiently to be evaluated. The best-performing model (Linear Regression, R² = 38.8%) leaves 61.2% of fare variance unexplained; complete model comparison metrics, prediction summaries, and visualizations are detailed in the Appendix (sections A.2–A.5).

## Conclusion

The results do not support the hypothesis that sustained increases in airline load factors above historical norms lead to disproportionate increases in average airfares, even when fuel prices and demand are held constant. Four findings inform this assessment.

**1. Load factor's association with fare is weak and, under controls, negatively signed.** Load factor's Pearson correlation with fare is 0.073, placing it 16th among 30 features and accounting for approximately 0.5% of fare variance. In the linear model, after controlling for 18 other features, the load factor coefficient is −$7.15, indicating that higher load factor is associated with modestly lower fares rather than higher ones. All four load factor variants exhibit this same negative coefficient.

**2. Fare does not increase disproportionately with load factor.** The quartile analysis shows mean fares of $256.66 in Q1, $251.20 in Q2, $279.44 in Q3, and $305.15 in Q4. The second quartile has the lowest mean fare, and the only notable increase coincides with the ≥80% saturation threshold. The `is_saturated` flag contributes +$11.77 after controls, a modest threshold effect that does not constitute the continuous disproportionate pattern described by the hypothesis.

**3. Fuel price and demand do not vary sufficiently to serve as controls.** Fuel price is constant at $2.2231 (std = 0.0000) across all 2,849 observations, yielding undefined correlations and a zero coefficient. Demand (`passengers_db1b`) has a correlation of 0.0023 with fare. The hypothesis requires that the relationship hold after accounting for these factors, but neither variable provides enough variation within a single-quarter cross-section to function as a meaningful control.

**4. The temporal dimension of the hypothesis cannot be assessed.** With only Q1 2025 in the dataset, there are no prior periods against which to evaluate whether load factor increases are sustained or whether their effects accumulate over time. The lag and rolling load factor features reduce to the current period value because no earlier observations exist.

The data instead indicate that fares are most strongly associated with route distance (r = 0.553, approximately 30.6% of variance) and operational disruption variables, which collectively carry 375 times more absolute coefficient weight than load factor in the linear model. Competition exerts modest downward pressure at −$8.04 per additional carrier. The best model (Linear Regression, R² = 38.8%) leaves 61.2% of fare variance unexplained.

Expanding the dataset to multiple quarters or years would allow the temporal component of the hypothesis to be tested and would introduce variance in fuel prices across periods. Incorporating nonlinear methods could also be considered, though the current R² and the limited contribution of load factor suggest that broader data coverage is likely to yield more informative results than increased model complexity. Within the scope of this analysis — 2,849 routes, 19 features, a single quarter — the methodology produced interpretable and reproducible results that do not provide evidence for the hypothesized relationship.

## References

- DB1B: https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FHK&QO_fu146_anzr=b4vtv0%20n0q%20Qr56v0n6v10%20f748rB
- T100: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM&QO_fu146_anzr=Nv4%20Pn44vr45
- Flight Delays: https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?type=21&pn=1
- Fuel Prices: https://fred.stlouisfed.org/series/DJFUELUSGULF
- Competition (Airline Count): https://rowzero.com/datasets/us-flights-dataset

---

## Appendix: Additional Results and Code

### A.1 Route Distance Comparison

Market distance quartiles (derived from `Outputs/Analysis_Table.csv`) exhibit a monotonic fare progression, illustrating the degree to which route distance is associated with fare levels:

| Distance Quartile | Mean Fare | Median Fare | Routes |
|---|---|---|---|
| Q1 (shortest routes) | $227.17 | $225.77 | 713 |
| Q2 | $255.94 | $247.53 | 712 |
| Q3 | $281.29 | $276.90 | 712 |
| Q4 (longest routes) | $333.76 | $315.94 | 712 |

From the shortest to the longest distance quartile, mean fares increase by $106.59 (46.9%), compared to the $39.52 (15.1%) raw difference between saturated and non-saturated routes. Distance accounts for approximately 30.6% of fare variance (0.5532² = 0.306), while load factor accounts for approximately 0.5% (0.0725² = 0.005).

### A.2 Full Model Comparison

Evaluated on 570 test samples (mean actual fare $269.43). Source: `Outputs/Evaluation/Metrics.csv`.

| Metric | Linear Regression | PCA Regression | Linear Advantage |
|---|---|---|---|
| RMSE | $65.04 | $68.14 | $3.10 lower |
| MAPE | 21.59% | 23.35% | 1.76pp lower |
| R² | 38.80% | 32.84% | 5.97pp higher |
| SNR | 17.16 | 15.64 | 1.53 higher |
| Accuracy (1−MAPE) | 78.41% | 76.65% | 1.76pp higher |

Linear Regression outperforms PCA Regression on each metric. The MAE of $45.14 corresponds to 16.7% of the mean fare, the median absolute error is $31.78, and the 90th percentile error is $100.46. The largest single prediction error is $419.36. Both models exhibit a slight positive bias in predicted values, with mean residuals of −$5.83 and −$6.42 respectively. Prediction details are from `Outputs/Modeling/Predictions_Summary.csv`; model comparison metrics are from `Outputs/Evaluation/Metrics.csv`; the bar chart visualization is `Outputs/Evaluation/Metrics_Comparison.png`.

### A.3 Prediction Summary

Source: `Outputs/Modeling/Predictions_Summary.csv`.

| Metric | Linear Regression | PCA Regression |
|---|---|---|
| Samples | 570 | 570 |
| Mean Actual Fare | $269.43 | $269.43 |
| Mean Predicted Fare | $275.26 | $275.85 |
| MAE | $45.14 | $49.32 |
| Median AE | $31.78 | $37.52 |
| P90 AE | $100.46 | $110.14 |
| Max AE | $419.36 | $410.11 |
| Mean Residual | −$5.83 | −$6.42 |

The PCA model compresses the data into 10 principal components, with PC1 dominated by load factor (see `Outputs/Evaluation/PCA_Variance.csv`). It performs worse than the unreduced Linear model on each error metric, indicating that the variance structure captured by PCA — which is largely organized around load factor — does not align closely with fare variation.

### A.4 PCA Variance Decomposition

Source: `Outputs/Evaluation/PCA_Variance.csv`.

| Component | Variance Explained | Cumulative | Top Feature | Loading |
|---|---|---|---|---|
| PC1 | 29.57% | 29.57% | `load_factor` | 0.394 |
| PC2 | 17.42% | 46.98% | `route_nas_delay_share` | 0.543 |
| PC3 | 13.54% | 60.52% | `competition_flights` | 0.339 |
| PC4 | 11.69% | 72.21% | — | — |
| PC5 | 7.04% | 79.24% | — | — |
| PC6–PC10 | 17.46% | 96.70% | — | — |

Load factor has the largest loading on PC1 (0.394), making it the most prominent variable in the overall covariance structure. However, PCA Regression (R² = 32.8%) yields 5.97 percentage points less explained variance than Linear Regression (R² = 38.8%), suggesting that the dimensions load factor dominates capture data structure rather than fare-level variation. The PCA cumulative variance curve is shown in the "PCA Cumulative Variance" subplot of `Outputs/Evaluation/Diagnostics.png`.

### A.5 Full Feature Importance Table (Linear Regression)

Source: `Outputs/Evaluation/Linear_Importance.csv`.

| Rank | Feature | Coefficient | Abs Coefficient |
|---|---|---|---|
| 1 | `route_weather_delay_share` | +4,437.72 | 4,437.72 |
| 2 | `route_avg_arr_delay_rate` | −2,645.41 | 2,645.41 |
| 3 | `route_nas_delay_share` | +1,381.65 | 1,381.65 |
| 4 | `route_avg_cancel_rate` | −1,054.74 | 1,054.74 |
| 5 | `route_late_aircraft_delay_share` | +757.18 | 757.18 |
| 6 | `route_carrier_delay_share` | +452.41 | 452.41 |
| 7 | `competition_cancel_rate` | +58.82 | 58.82 |
| 8 | `is_saturated` | +11.77 | 11.77 |
| 9 | `competition_unique_carriers` | −8.04 | 8.04 |
| 10 | `route_avg_load_factor` | −7.15 | 7.15 |
| 11 | `load_factor` | −7.15 | 7.15 |
| 12 | `rolling_load_factor_2` | −7.15 | 7.15 |
| 13 | `lag_load_factor` | −7.15 | 7.15 |
| 14 | `competition_delay15_rate` | −4.10 | 4.10 |
| 15 | `competition_avg_delay` | −0.09 | 0.09 |
| 16 | `market_distance` | +0.07 | 0.07 |
| 17 | `competition_flights` | +0.02 | 0.02 |
| 18 | `passengers_db1b` | −0.001 | 0.001 |
| 19 | `avg_fuel_price` | 0.00 | 0.00 |

### A.6 Full Correlation with Average Fare (Top 15)

Source: full correlation matrix computed in `Model.py` via `run_modeling_pipeline()`, top correlations also summarized in `Outputs/Evaluation/Conclusions.csv`.

| Feature | Pearson r with avg_fare |
|---|---|
| `market_distance` | 0.5532 |
| `is_saturated` | 0.2191 |
| `route_late_aircraft_delay_share` | −0.1249 |
| `dest_late_aircraft_delay_share` | −0.1222 |
| `origin_late_aircraft_delay_share` | −0.1208 |
| `competition_unique_carriers` | −0.1138 |
| `competition_flights` | −0.1009 |
| `departures_performed` | −0.0874 |
| `dest_arr_delay_rate` | 0.0870 |
| `origin_cancel_rate` | 0.0867 |
| `route_avg_arr_delay_rate` | 0.0846 |
| `origin_weather_delay_share` | 0.0841 |
| `route_avg_cancel_rate` | 0.0796 |
| `route_weather_delay_share` | 0.0777 |
| `origin_arr_delay_rate` | 0.0774 |
| `load_factor` | 0.0725 |

### A.7 Focused Correlation Matrix

Source: `Outputs/Modeling/Correlation.csv`.

|  | load_factor | competition_unique_carriers | route_avg_arr_delay_rate | avg_fuel_price | passengers_db1b | avg_fare |
|---|---|---|---|---|---|---|
| load_factor | 1.000 | 0.350 | 0.119 | NaN | 0.207 | 0.074 |
| competition_unique_carriers | 0.350 | 1.000 | 0.053 | NaN | 0.646 | −0.114 |
| route_avg_arr_delay_rate | 0.119 | 0.053 | 1.000 | NaN | 0.054 | 0.129 |
| avg_fuel_price | NaN | NaN | NaN | NaN | NaN | NaN |
| passengers_db1b | 0.207 | 0.646 | 0.054 | NaN | 1.000 | 0.002 |
| avg_fare | 0.074 | −0.114 | 0.129 | NaN | 0.002 | 1.000 |

### A.8 Source Code Files

| File | Description |
|---|---|
| `Main.py` | Entry point. Orchestrates the full pipeline: calls data preprocessing, model training, and evaluation in stage order. |
| `Data.py` | Data preprocessing. Loads raw CSV files from five data sources (DB1B, T-100, Flight Delays, Fuel Prices, Competition), cleans and filters to Q1 2025 and TX/GA/CA routes, merges into a unified route-quarter analysis table with 19 features, and saves `Analysis_Table.csv` to `Outputs/`. |
| `Model.py` | Modeling and evaluation. Builds the model frame with load factor context features (lag, rolling, saturation flag), runs correlation analysis, trains Linear Regression and PCA Regression, computes RMSE/MAPE/R²/SNR/Accuracy metrics, generates all visualizations (heatmaps, scatter plots, diagnostics, model comparison panels), and saves all output files to `Outputs/Modeling/` and `Outputs/Evaluation/`. |
