# Streamlit Deployment

## Local run

From the project root, install the dependencies and start the app.

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud

1. Ensure the GitHub repository contains the required project files, including `data/processed`, `models`, and `reports`.
2. Do not upload the local `.vendor` directory.
3. In Streamlit Community Cloud, create a new app from the GitHub repository.
4. Set the main file path to `app/streamlit_app.py`.
5. Deploy. The platform installs the dependencies from `requirements.txt` and loads the already-trained models. It does not retrain models at startup.

## Required tracked files

Keep these files in the GitHub repository:

- `data/processed/master_dataset.csv`
- `models/regression/rating_model.pkl`
- `models/classification/visit_mode_model.pkl`
- `models/recommendation/popular_recommendations.csv`
- `reports/eda/business_insights.json`

The raw Excel files are not included in the GitHub repository because they are not required to run the deployed app.
