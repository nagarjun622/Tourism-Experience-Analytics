from pathlib import Path
import pandas as pd, joblib
ROOT=Path(__file__).resolve().parents[1]
def load_assets():
    return (pd.read_csv(ROOT/'data/processed/master_dataset.csv'),joblib.load(ROOT/'models/regression/rating_model.pkl'),joblib.load(ROOT/'models/classification/visit_mode_model.pkl'),pd.read_csv(ROOT/'models/recommendation/popular_recommendations.csv'))
def recommendation_table(popular, attraction_type=None, city=None, n=10):
    x=popular.copy()
    if attraction_type: x=x[x.AttractionType==attraction_type]
    if city: x=x[x.AttractionCity==city]
    return x.head(n)
