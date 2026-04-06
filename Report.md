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

Airfare pricing is shaped by a combination of operational, economic, and market factors, making it a useful topic for both industry analysis and consumer welfare research. According to Wensveen (2023), who provides a comprehensive overview of the global airline industry, airfare outcomes are the product of interacting supply-side costs, demand conditions, and regulatory structures. In the U.S. domestic market specifically, Williams (2022) demonstrates through a structural model of dynamic airline pricing that carriers set fares based on remaining seat availability, stochastic demand, and competitive pressure — establishing that capacity management, route-level competition, fuel costs, and operational disruptions each play measurable roles in determining what passengers pay.

A central indicator of how airlines balance capacity and demand is the load factor, defined as the share of available seats that are sold on a given flight. According to Cramer and Thams (2021), load factor is a core metric in airline revenue management because it reflects how effectively carriers convert available capacity into revenue. Their analysis of revenue management strategy shows that airlines often pursue high load factors through discounted booking classes, meaning that a high load factor does not necessarily correspond to high per-seat revenue. In a related study, Zou et al. (2025) examine load factor dynamics in domestic air markets and find that route distance and airport size are the primary structural drivers of both load factors and ticket prices, with carriers routinely operating above 80% capacity on established routes. This raises an important empirical question. When flights operate near capacity on a sustained basis, do airlines obtain greater pricing power, or do competitive and cost constraints limit fare increases?

The competitive dimension of this question is informed by Ng et al. (2022), who analyze airline yield and competition in the Japanese domestic market during the COVID-19 pandemic. Their findings show that market concentration and competitive dynamics shape pricing independently of capacity utilization, suggesting that competition may constrain fare increases even when load factors are high. Together, these studies frame the empirical tension that motivates the project's primary hypothesis:

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

### Results

The results below address the three testable components of the hypothesis: whether load factor correlates with fare, whether that relationship is disproportionate, and whether it persists after accounting for fuel prices and demand.

#### Correlation with Fare

Load factor's Pearson correlation with average fare is r = 0.073, ranking 16th among 30 numeric features and accounting for roughly 0.5% of fare variance (source: `Outputs/Evaluation/Conclusions.txt`; full ranking in Appendix A.6). Market distance leads at r = 0.553 (≈30.6% of variance), followed by the binary `is_saturated` flag at r = 0.219. The hypothesis requires a positive directional association between load factor and fare; a correlation this weak does not establish one.

The focused correlation matrix (source: `Outputs/Modeling/Correlation.txt`) also reveals that fuel price produces NaN correlations due to zero within-quarter variance, and passenger demand correlates at 0.002 with fare — effectively zero. Both of the hypothesis's required control variables are inert in this single-quarter design.

#### Regression Coefficients and Disproportionality

The linear model assigns load factor a coefficient of −$7.15, meaning that after holding 18 other features constant, a one-unit increase in load factor is associated with a $7.15 *decrease* in fare — opposite to the hypothesis direction (source: `Outputs/Evaluation/Linear_Importance.txt`; full table in Appendix A.5). All four load factor variants share this same negative coefficient.

The `is_saturated` flag (≥80% load factor) receives a coefficient of +$11.77, suggesting a modest discrete fare premium at high capacity levels. However, the quartile analysis below shows that this effect is confined to Q4 rather than accumulating progressively across the load factor range:

| Load Factor Quartile | Mean Fare | Median Fare | Routes |
|---|---|---|---|
| Q1 (lowest) | $256.66 | $252.06 | 682 |
| Q2 | $251.20 | $243.74 | 681 |
| Q3 | $279.44 | $271.73 | 681 |
| Q4 (highest) | $305.15 | $298.22 | 681 |

Q2 has the lowest mean fare, falling below Q1 and breaking the monotonic pattern a disproportionate relationship would require. The only notable jump occurs between Q3 and Q4, where saturation is prevalent. This pattern indicates a threshold effect at high capacity, not the continuous disproportionate increase the hypothesis describes.

#### Model Comparison

Linear Regression (R² = 38.80%, RMSE = $65.04) outperforms PCA Regression (R² = 32.84%, RMSE = $68.14) on every evaluation metric (source: `Outputs/Evaluation/Metrics.txt`; full comparison in Appendix A.2). PCA compresses the 19 features into 10 principal components capturing 96.70% of total variance, with PC1 dominated by load factor (loading = 0.394, 29.57% variance explained; source: `Outputs/Evaluation/PCA_Variance.txt`). The fact that the PCA model — which organizes data primarily around load factor's variance — performs worse confirms that load factor's dominance in the covariance structure does not translate into fare prediction.

*Summary.* Correlation is near zero, the controlled coefficient is negative, the quartile pattern is non-monotonic, and the hypothesis's control variables have no within-sample variation. None of the three techniques produces evidence supporting the hypothesized relationship.

### Data Visualization

#### Fig 1.1 — Modeling Overview

![Fig 1.1: Modeling Overview](Outputs/Modeling/Modeling_Overview.png)

*Fig 1.1: Four-panel summary of feature relationships. Top-left: correlation heatmap showing the block of highly correlated load factor variants and fare's weak association with most predictors. Top-right: Load Factor vs. Airfare scatter plot, displaying widely dispersed points with no upward trend. Bottom-left: Competition vs. Airfare, showing a mild negative spread as carrier count increases. Bottom-right: Delay Rate vs. Airfare, showing a loose positive association.*

The heatmap visually isolates the load factor cluster — four variants correlating near 1.0 with each other but registering only faint color against fare. This confirms why the linear model assigns them identical coefficients: they carry redundant information. The Load Factor vs. Airfare scatter is the most directly relevant panel. If the hypothesis held, points would trend upward with increasing load factor; instead, the cloud is flat. By contrast, the Delay Rate scatter shows a visible positive spread, consistent with delay features dominating the regression coefficients.

#### Fig 1.2 — Actual vs. Predicted Fares

![Fig 1.2: Actual vs. Predicted](Outputs/Evaluation/Actual_Vs_Predicted.png)

*Fig 1.2: Predicted versus actual fares for Linear Regression and PCA Regression on 570 test samples. The diagonal line represents perfect prediction.*

Both models cluster around the diagonal at moderate fare levels ($150–$350) but diverge at higher fares. The Linear model's mean absolute error is $45.14 (16.7% of the $269.43 mean fare), with errors widening beyond $400 where the model systematically under-predicts (source: `Outputs/Modeling/Predictions_Summary.txt`). PCA Regression's scatter is visibly more dispersed, consistent with its higher RMSE. The under-prediction at high fare levels reflects the dominance of distance and delay variables in this range — features that the PCA compression partially collapses — rather than any load factor effect.

#### Fig 1.3 — Diagnostics

![Fig 1.3: Diagnostics](Outputs/Evaluation/Diagnostics.png)

*Fig 1.3: Residual histogram, Q-Q plot, and PCA cumulative variance curve.*

The residual histogram is approximately symmetric around zero with moderate tails, indicating no systematic directional bias. The Q-Q plot shows approximate normality in the core with departures in the tails, expected given airfare heterogeneity. The PCA cumulative variance curve rises steeply through PC1–PC3 (60.5%) then gradually plateaus. This shape illustrates that load factor and delay features account for most data variance, but since PCA Regression underperforms Linear Regression, the dimensions these features dominate do not correspond closely to fare variation.

#### Fig 1.4 — Metrics Comparison

![Fig 1.4: Metrics Comparison](Outputs/Evaluation/Metrics_Comparison.png)

*Fig 1.4: Side-by-side bar chart of RMSE, MAPE, R², SNR, and Accuracy (1−MAPE) for Linear Regression versus PCA Regression.*

Linear Regression leads on every metric. The R² gap (38.8% vs. 32.8%) is the most informative: compressing correlated features through PCA reduces predictive power rather than improving it. This rules out the possibility that multicollinearity among load factor variants artificially inflated the unreduced model's performance, and it reinforces that the variance load factor dominates is not fare-relevant variance.

### Analysis

The results and visualizations above address each component of the hypothesis individually. This section synthesizes them into a unified assessment.

**Load factor's association with fare is negligible.** At r = 0.073, load factor explains roughly 0.5% of fare variance — a result confirmed both numerically by the correlation analysis and visually by the flat scatter in Fig 1.1. When other features are held constant, load factor's coefficient reverses sign to −$7.15. The hypothesis requires a positive, meaningful relationship; neither the raw association nor the controlled estimate provides one.

**The fare pattern across load factor levels is not disproportionate.** A disproportionate effect would produce progressively larger fare increases at higher load factor levels. Instead, Q2 has the lowest mean fare, and the only significant increase occurs in Q4, where it coincides with the ≥80% saturation threshold. The linear model isolates this as a discrete +$11.77 premium — a modest threshold effect, not a continuous escalation. Fig 1.2 reinforces this: the models' prediction errors at high fare levels are driven by distance and delay features, not by capacity pressure.

**The hypothesis's control conditions cannot be evaluated.** Fuel price is constant across all 2,849 observations (std = 0.0000), yielding NaN correlations and a zero regression coefficient. Passenger demand correlates at 0.002 with fare. With neither variable exhibiting within-sample variation, the hypothesis condition "even when fuel prices and demand are held constant" lacks a testable contrast in this single-quarter cross-section.

**Load factor dominates data structure but not fare structure.** PCA identifies load factor as the top-loading feature on PC1 (29.6% of total variance), yet PCA Regression explains less fare variance than the unreduced Linear model. This dissociation, visible in the cumulative variance curve of Fig 1.3 and the metric gaps in Fig 1.4, demonstrates that the dimensions load factor organizes are not the dimensions along which fares vary. Operational disruption variables and route distance — features that rank lower in overall covariance — carry far more predictive weight for fares.

**What the data indicate instead.** Fares are most strongly associated with route distance (r = 0.553) and delay-related features, which collectively hold over 375 times more absolute coefficient weight than load factor. Competition exerts a modest downward effect (−$8.04 per additional carrier). Together, these features account for the 38.8% of fare variance the best model explains, while the remaining 61.2% lies outside the scope of the available variables. Load factor, despite being the most prominent feature in the data's covariance structure, does not contribute meaningfully to this explained portion.

## Conclusion

The analysis does not provide evidence for the hypothesis. Load factor's correlation with fare is near zero, its controlled coefficient is negative, the quartile pattern is non-monotonic, and the required control variables lack sufficient variation for evaluation. The single-quarter design further prevents assessment of the temporal dimension.

These results suggest that airfare in this dataset is shaped primarily by route distance and operational disruption measures, with competition exerting modest downward pressure. Load factor's prominence in the data's covariance structure — confirmed by PCA — does not translate into fare-level explanatory power.

Two directions would strengthen future work. First, expanding the dataset to multiple quarters or years would introduce temporal variation in both load factor and fuel prices, enabling a direct test of the "sustained over time" condition. Second, nonlinear or threshold-based models could better capture the discrete saturation effect identified in this analysis, though the current R² (38.8%) and load factor's minimal contribution suggest that broader data coverage is likely to be more productive than increased model complexity.

## References

### Data Sources

- DB1B: https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FHK&QO_fu146_anzr=b4vtv0%20n0q%20Qr56v0n6v10%20f748rB
- T100: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM&QO_fu146_anzr=Nv4%20Pn44vr45
- Flight Delays: https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?type=21&pn=1
- Fuel Prices: https://fred.stlouisfed.org/series/DJFUELUSGULF
- Competition (Airline Count): https://rowzero.com/datasets/us-flights-dataset

### Academic Sources

- Cramer, C., & Thams, A. (2021). *Airline Revenue Management*. Springer. https://doi.org/10.1007/978-3-658-33721-6
- Ng, K. T., Fu, X., Hanaoka, S., & Oum, T. H. (2022). Japanese aviation market performance during the COVID-19 pandemic — Analyzing airline yield and competition in the domestic market. *Transport Policy*, 116, 139–151. https://doi.org/10.1016/j.tranpol.2021.12.007
- Wensveen, J. (2023). *Air Transportation: A Global Management Perspective* (9th ed.). Routledge. https://doi.org/10.4324/9780429346156
- Williams, K. R. (2022). The welfare effects of dynamic pricing: Evidence from airline markets. *Econometrica*, 90(2), 831–858. https://doi.org/10.3982/ECTA16180
- Zou, W., Zhang, Z., Gao, S., Wang, L., & Jiang, Y. (2025). Drivers of short-term essential air service operations: Load factor, pricing, and subsidy policies in China's domestic air market. *Transport Policy*, 150, 103–118. https://doi.org/10.1016/j.tranpol.2024.12.017

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

Evaluated on 570 test samples (mean actual fare $269.43). Source: `Outputs/Evaluation/Metrics.txt`.

| Metric | Linear Regression | PCA Regression | Linear Advantage |
|---|---|---|---|
| RMSE | $65.04 | $68.14 | $3.10 lower |
| MAPE | 21.59% | 23.35% | 1.76pp lower |
| R² | 38.80% | 32.84% | 5.97pp higher |
| SNR | 17.16 | 15.64 | 1.53 higher |
| Accuracy (1−MAPE) | 78.41% | 76.65% | 1.76pp higher |

Linear Regression outperforms PCA Regression on each metric. The MAE of $45.14 corresponds to 16.7% of the mean fare, the median absolute error is $31.78, and the 90th percentile error is $100.46. The largest single prediction error is $419.36. Both models exhibit a slight positive bias in predicted values, with mean residuals of −$5.83 and −$6.42 respectively. Prediction details are from `Outputs/Modeling/Predictions_Summary.txt`; model comparison metrics are from `Outputs/Evaluation/Metrics.txt`; the bar chart visualization is `Outputs/Evaluation/Metrics_Comparison.png`.

### A.3 Prediction Summary

Source: `Outputs/Modeling/Predictions_Summary.txt`.

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

The PCA model compresses the data into 10 principal components, with PC1 dominated by load factor (see `Outputs/Evaluation/PCA_Variance.txt`). It performs worse than the unreduced Linear model on each error metric, indicating that the variance structure captured by PCA — which is largely organized around load factor — does not align closely with fare variation.

### A.4 PCA Variance Decomposition

Source: `Outputs/Evaluation/PCA_Variance.txt`.

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

Source: `Outputs/Evaluation/Linear_Importance.txt`.

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

Source: full correlation matrix computed in `Model.py` via `run_modeling_pipeline()`, top correlations also summarized in `Outputs/Evaluation/Conclusions.txt`.

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

Source: `Outputs/Modeling/Correlation.txt`.

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
