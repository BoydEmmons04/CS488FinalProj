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

## Core Pipeline Files

```text
main.py        # Orchestrates full pipeline
ingest.py      # Data preprocessing and cleaned CSV generation
features.py    # Feature construction and analysis table build
modeling.py    # Correlation analysis + Linear Regression + PCA
evaluation.py  # Metrics and project-aligned visual outputs
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

1. `outputs/modeling/correlation_focus.txt`
2. `outputs/modeling/modeling_overview_panel.png`
3. `outputs/modeling/test_predictions_summary.txt`

Evaluation outputs:

1. `outputs/evaluation/model_metrics.txt`
2. `outputs/evaluation/model_comparison_panel.png`
3. `outputs/evaluation/actual_vs_predicted_panel.png`
4. `outputs/evaluation/data_diagnostics_panel.png`
5. `outputs/evaluation/feature_importance_linear.txt`
6. `outputs/evaluation/pca_explained_variance.txt`
7. `outputs/evaluation/time_series_summary.txt`
8. `outputs/evaluation/conclusions_summary.txt`
9. `outputs/evaluation/report_summary_table.txt`
