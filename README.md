# Tourism Experience Analytics

An end-to-end tourism analytics project that transforms visitor interactions into three practical capabilities: attraction rating prediction, visit-mode classification, and attraction recommendations.

The project is designed for portfolio presentation, academic evaluation, technical interviews, and Streamlit deployment.

## Project objectives

- Predict an attraction rating on the 1 to 5 scale.
- Classify a visitor's travel mode: Business, Couples, Family, Friends, or Solo.
- Recommend attractions using interaction volume, ratings, attraction type, and location.
- Present the results through a reusable Streamlit application.

## Dataset

The supplied relational tourism dataset contains transaction, user, attraction, location, attraction-type, and visit-mode tables.

The final integrated dataset contains **52,930 interactions**, **33,530 users**, and **30 attractions**. User geography and attraction geography are retained as separate fields so that tourist origin is never confused with attraction location.

## Project structure

```text
Tourism-Experience-Analytics/
├── app/                         # Streamlit application
├── data/
│   ├── raw/                     # Original Excel files, never overwritten
│   └── processed/               # Cleaned tables and master dataset
├── models/                      # Saved machine-learning and recommendation artifacts
├── notebooks/                   # Analysis notebooks and complete Colab notebook
├── reports/                     # Data quality reports, EDA figures, and model reports
├── presentation/                # Presentation outline
├── run_pipeline.py              # Reproducible end-to-end pipeline
├── requirements.txt             # Application and analysis dependencies
├── DEPLOYMENT.md                # Streamlit deployment instructions
└── README.md
```

## Data preparation

Raw Excel workbooks remain unchanged in `data/raw`.

Before modeling, the pipeline profiles all source tables for schema, data types, missing values, duplicates, key uniqueness, invalid ranges, and foreign-key consistency. It removes only exact duplicate records and documents the before/after counts, affected rows, and reason in `reports/data_quality/cleaning_log.csv`.

`Updated_Item.xlsx` is examined as a supplemental source and is not automatically merged with `Item.xlsx`.

## Exploratory analysis

The EDA covers:

- Univariate analysis of ratings, visit modes, years, months, and attraction types.
- Bivariate analysis of ratings by visit mode, attraction-type demand, and seasonal visit modes.
- Multivariate analysis of attraction type, visit mode, demand volume, and average rating.

Saved figures and tables are available in `reports/eda/`.

## Model results

| Component | Selected approach | Evaluation result |
|---|---|---|
| Rating prediction | Random Forest Regressor | MAE: 0.737, RMSE: 0.950, R2: 0.042 |
| Visit-mode classification | Class-balanced Random Forest | Accuracy: 0.357, Macro F1: 0.278 |
| Recommendations | Smoothed popularity baseline with content filters | Cold-start-safe ranking by interaction volume and average rating |

The rating model can provide a useful supporting estimate, although its low R2 indicates that available features explain only a limited share of rating variation. The classifier is evaluated with macro metrics and a confusion matrix because the visit-mode classes are imbalanced.

The recommendation system uses a popularity and rating-smoothing baseline because the user-attraction matrix is highly sparse. Attraction type and city filters provide more relevant recommendations, while the popularity ranking remains a reliable fallback for new users.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Recreate all processed datasets, reports, models, and notebooks:

```bash
python run_pipeline.py
```

Launch the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

## Streamlit application

The application contains the following pages:

- Project Overview
- Tourism Analytics
- Rating Prediction
- Visit Mode Prediction
- Attraction Recommendations
- Insights

It loads saved preprocessing pipelines and trained models, validates inputs, and does not retrain models while the application runs.

## Google Colab

Use `notebooks/Tourism_Experience_Analytics_Complete_Colab.ipynb` for a single end-to-end notebook demonstration. It supports direct upload of the supplied tourism dataset ZIP file and includes data preparation, all EDA levels, machine learning, and recommendations.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Community Cloud instructions. The deployed application requires the processed master dataset, saved model files, recommendation artifacts, and insights JSON file to remain in the repository.

## Limitations and future work

- Rating prediction performance is limited by the available behavioral and attraction features.
- Visit-mode classification needs richer visitor-profile features to improve minority-class performance.
- The interaction matrix is sparse, limiting the reliability of collaborative filtering.
- A time-ordered interaction history would enable stronger offline recommendation evaluation, such as Precision@K and Recall@K.

Future work can add behavioral features, attraction text embeddings, richer user profiles, and time-aware recommendation evaluation.
