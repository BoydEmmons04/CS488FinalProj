# Airline Load Factor and Airfare Analysis

This project tests whether higher airline load factor is associated with higher airfare, using three expert techniques:

1. Correlation Analysis
2. Linear Regression
3. PCA Regression (dimensionality reduction + regression)

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

## Core Pipeline Files

```text
main.py   # Orchestrates the full pipeline in stage order
data.py   # Stage 1 (preprocessing) + Stage 2 (feature engineering)
model.py  # Stage 3 (feature selection) + Stage 4 (expert techniques)
		  # + Stage 5 (evaluation metrics) + Stage 6 (visualizations)
```

## Pipeline Stages

| Stage | Description | File |
|-------|-------------|------|
| 1 | Data Preprocessing — load raw CSVs, filter to Q1 2025 and target airports, write cleaned tables | data.py |
| 2 | Feature Engineering — merge sources, compute load factor, delay shares, is_saturated flag | data.py |
| 3 | Feature Selection — add lag/rolling load features, impute medians, fix 19-feature set | model.py |
| 4 | Expert Techniques — Correlation Analysis, Linear Regression, PCA Regression | model.py |
| 5 | Performance Evaluation — RMSE, MAPE, R², SNR, prediction accuracy | model.py |
| 6 | Data Visualization — heatmaps, histograms, scatter plots, accuracy panels | model.py |

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

1. `outputs/modeling/correlation.txt`
2. `outputs/modeling/modeling_overview.png`
3. `outputs/modeling/predictions_summary.txt`

Evaluation outputs:

1. `outputs/evaluation/metrics.txt`
2. `outputs/evaluation/metrics_comparison.png`
3. `outputs/evaluation/actual_vs_predicted.png`
4. `outputs/evaluation/diagnostics.png`
5. `outputs/evaluation/linear_importance.txt`
6. `outputs/evaluation/pca_variance.txt`
7. `outputs/evaluation/conclusions.txt`
