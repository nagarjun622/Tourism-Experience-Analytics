# Tourism Experience Analytics

End-to-end analysis of tourism interactions with rating prediction, visit-mode classification, and attraction recommendations.

## Quick start

```bash
python run_pipeline.py
streamlit run app/streamlit_app.py
```

## Data decisions

Raw Excel files remain untouched in `data/raw`. The project removes only exact duplicates and records the before/after counts in `reports/data_quality/cleaning_log.csv`. `Updated_Item` is assessed separately and is not automatically merged.

## Modeling

Regression predicts `Rating` with MAE, MSE, RMSE, and R2. Classification predicts `VisitMode` and reports macro precision, recall, F1, and a confusion matrix. Recommendations use a smoothed popularity baseline plus attraction content filters; this provides a reliable cold-start fallback for sparse interaction data.

## Results from the supplied data

- Rating model: random forest; MAE 0.737, RMSE 0.950, R2 0.042.
- Visit mode model: class-balanced random forest; accuracy 0.357, macro F1 0.278.
- Recommendation: popularity and rating smoothing with content filters. The approach is chosen because sparse interactions make a simple user-neighbor model unstable.

See `reports/Project_Report.md` for limitations and `presentation/Presentation_Outline.md` for the presentation narrative.
