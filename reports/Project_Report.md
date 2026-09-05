# Tourism Experience Analytics Report

## Purpose

This project analyzes tourism interactions and provides three functions: rating prediction, visit-mode classification, and attraction recommendations.

## Data and quality controls

The supplied Excel workbooks remain unchanged under `data/raw`. The pipeline audits schemas, missing values, duplicates, primary keys, and foreign keys. It writes dataset profiles and relationship checks to `reports/data_understanding`.

Only exact duplicate records are removed. `reports/data_quality/cleaning_log.csv` records before and after counts, rows affected, and the reason for every cleaning action. `Updated_Item` is assessed separately and retained as a supplemental source; it is not automatically merged with `Item`.

## Integration

The master dataset joins transactions to user and attraction dimensions. User geography and attraction geography are deliberately represented in separate columns to prevent an origin/destination mix-up. Merge validation confirms that transaction joins do not multiply rows.

## Modeling results

The rating model uses a random forest with persisted preprocessing. On the held-out test set it produced MAE 0.737, RMSE 0.950, MSE 0.902, and R2 0.042. The Streamlit application clips displayed rating predictions to the valid 1 to 5 scale.

The visit-mode model uses a class-balanced random forest. It reached accuracy 0.357, macro precision 0.322, macro recall 0.390, and macro F1 0.278. These results indicate that the supplied geographic and visit-time features have limited power to distinguish visit modes. The application displays probabilities, and the report retains class-level metrics and a confusion matrix.

## Recommendation approach

The recommender uses a smoothed popularity score based on interaction volume and average rating, with attraction type and location available for filtering. This is a deliberate cold-start-safe choice. The pipeline reports matrix sparsity and does not claim a collaborative-filtering evaluation when the available records lack a reliable chronological next-item holdout.

## How to run

Install dependencies from `requirements.txt`, then run `python run_pipeline.py` from the project root. Launch the interactive application with `streamlit run app/streamlit_app.py`.
