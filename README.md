# Airline Load Factor and Airfare Analysis

This project tests whether higher airline load factor is associated with higher airfare, using a core pipeline built around:

1. Correlation analysis
2. Linear Regression
3. PCA

## Requirements

1. Python 3.9+
2. Git

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the full pipeline:

```bash
python main.py
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

## Core Pipeline Files

```text
main.py        # Orchestrates full pipeline
ingest.py      # Data preprocessing and cleaned CSV generation
features.py    # Feature construction and analysis table build
modeling.py    # Correlation analysis + Linear Regression + PCA
evaluation.py  # Metrics and project-aligned visual outputs
dashboard.py   # Streamlit viewer for outputs
```

## Data Flow

1. Raw data in `Project Datasets/`:
	- Competition (Airline Count)
	- DB1B Market Airline Ticket Data
	- Flight Delays (Delay Cause)
	- Fuel Prices
	- T-100 (Load Factor)
2. Cleaned data written to `cleaned_data/`
3. Modeling artifacts written to `outputs/modeling/`
4. Evaluation artifacts written to `outputs/evaluation/`

## Output Artifacts (Current)

Modeling outputs:

1. `outputs/modeling/correlation_focus.csv`
2. `outputs/modeling/correlation_heatmap.png`
3. `outputs/modeling/load_factor_vs_airfare.png`
4. `outputs/modeling/competition_vs_airfare.png`
5. `outputs/modeling/delay_rate_vs_airfare.png`
6. `outputs/modeling/test_predictions.csv`

Evaluation outputs:

1. `outputs/evaluation/model_metrics.csv`
2. `outputs/evaluation/rmse_snr_comparison.png`
3. `outputs/evaluation/r2_mape_comparison.png`
4. `outputs/evaluation/accuracy_comparison.png`
5. `outputs/evaluation/actual_vs_predicted_linear_regression.png`
6. `outputs/evaluation/pca_explained_variance.png`
7. `outputs/evaluation/key_variable_histograms.png`
8. `outputs/evaluation/feature_importance_linear.csv`
9. `outputs/evaluation/pca_explained_variance.csv`
10. `outputs/evaluation/fare_load_factor_over_time.png`
11. `outputs/evaluation/fuel_price_over_time.png`
12. `outputs/evaluation/time_series_summary.csv`
13. `outputs/evaluation/feature_relationships_vs_fare.png`
14. `outputs/evaluation/delay_cause_composition.png`
