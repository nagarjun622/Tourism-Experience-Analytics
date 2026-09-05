import streamlit as st
import pandas as pd
from utils import load_assets, recommendation_table
st.set_page_config(page_title='Tourism Experience Analytics',layout='wide')
@st.cache_resource
def get_assets():
    return load_assets()
master, rating_model, mode_model, popular=get_assets()
st.title('Tourism Experience Analytics')
page=st.sidebar.radio('Navigate',['Project Overview','Tourism Analytics','Rating Prediction','Visit Mode Prediction','Attraction Recommendations','Insights'])
features=['VisitYear','VisitMonth','UserContinent','UserRegion','UserCountry','UserCity','AttractionType','AttractionCity','AttractionCountry','AttractionRegion','AttractionContinent']
def form_inputs():
    vals={}
    for c in features:
        values=sorted(master[c].dropna().unique())
        vals[c]=st.selectbox(c,values)
    vals['VisitYear'] = int(vals['VisitYear'])
    vals['VisitMonth'] = int(vals['VisitMonth'])
    return pd.DataFrame([vals])
if page=='Project Overview': st.write('An end-to-end project for rating prediction, visit-mode classification, and attraction recommendations. Models reuse the persisted training pipelines.')
elif page=='Tourism Analytics':
    st.bar_chart(master['VisitMode'].value_counts()); st.dataframe(master.groupby('Attraction').Rating.agg(['count','mean']).sort_values('count',ascending=False).head(15))
elif page=='Rating Prediction':
    x=form_inputs()
    if st.button('Predict rating'): st.metric('Predicted Rating',f'{rating_model.predict(x).clip(1,5)[0]:.2f} / 5')
elif page=='Visit Mode Prediction':
    x=form_inputs()
    if st.button('Predict visit mode'):
        st.success(mode_model.predict(x)[0]); probs=mode_model.predict_proba(x)[0]; st.bar_chart(dict(zip(mode_model.classes_,probs)))
elif page=='Attraction Recommendations':
    typ=st.selectbox('Optional attraction type',['']+sorted(master.AttractionType.dropna().unique().tolist())); city=st.selectbox('Optional attraction city',['']+sorted(master.AttractionCity.dropna().unique().tolist())); st.dataframe(recommendation_table(popular,typ or None,city or None)[['Attraction','AttractionType','AttractionCity','AttractionCountry','recommendation_score','reason']])
else:
    import json
    from pathlib import Path

    insights_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "eda"
        / "business_insights.json"
    )

    with open(insights_path, "r", encoding="utf-8") as f:
        insights = json.load(f)

    st.header("📊 Business Insights")
    st.write(
        "Key observations and business implications derived from the "
        "tourism analytics and machine learning analysis."
    )

    for i, insight in insights.items():
        st.subheader(f"Insight {i}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔎 Observation")
            st.info(insight["observation"])

        with col2:
            st.markdown("### 💡 Business Insight")
            st.success(insight["business_insight"])

        st.markdown("### 📌 Business Implication")
        st.write(insight["business_implication"])

        st.divider()