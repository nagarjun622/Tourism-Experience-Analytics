import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import load_assets, recommendation_table


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)


# ---------------------------------------------------------
# Load project assets
# ---------------------------------------------------------
@st.cache_resource
def get_assets():
    return load_assets()


master, rating_model, mode_model, popular = get_assets()


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------
st.title("🌍 Tourism Experience Analytics")


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Tourism Analytics",
        "Rating Prediction",
        "Visit Mode Prediction",
        "Attraction Recommendations",
        "Insights"
    ]
)


# ---------------------------------------------------------
# Prediction input features
# ---------------------------------------------------------
features = [
    "VisitYear",
    "VisitMonth",
    "UserContinent",
    "UserRegion",
    "UserCountry",
    "UserCity",
    "AttractionType",
    "AttractionCity",
    "AttractionCountry",
    "AttractionRegion",
    "AttractionContinent"
]


# ---------------------------------------------------------
# Common input form
# ---------------------------------------------------------
def form_inputs():
    vals = {}

    for c in features:
        values = sorted(master[c].dropna().unique())
        vals[c] = st.selectbox(c, values)

    vals["VisitYear"] = int(vals["VisitYear"])
    vals["VisitMonth"] = int(vals["VisitMonth"])

    return pd.DataFrame([vals])


# ---------------------------------------------------------
# Project Overview
# ---------------------------------------------------------
if page == "Project Overview":

    st.header("📌 Project Overview")

    st.write(
        "An end-to-end tourism analytics project for rating prediction, "
        "visit-mode classification, and attraction recommendations. "
        "The application uses persisted machine learning models and "
        "processed tourism interaction data."
    )

    st.subheader("Key Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ⭐ Rating Prediction")
        st.write(
            "Predict the expected attraction rating on a scale of 1 to 5 "
            "based on tourism interaction features."
        )

    with col2:
        st.markdown("### 👥 Visit Mode Prediction")
        st.write(
            "Classify the likely visit mode such as Couples, Family, "
            "Friends, Solo, or Business."
        )

    with col3:
        st.markdown("### 🗺️ Attraction Recommendations")
        st.write(
            "Recommend attractions using popularity and attraction "
            "characteristics as a cold-start recommendation strategy."
        )


# ---------------------------------------------------------
# Tourism Analytics
# ---------------------------------------------------------
elif page == "Tourism Analytics":

    st.header("📊 Tourism Analytics")

    st.subheader("Visit Mode Distribution")

    visit_mode_counts = master["VisitMode"].value_counts()

    st.bar_chart(visit_mode_counts)

    st.subheader("Most Frequently Visited Attractions")

    attraction_summary = (
        master.groupby("Attraction")
        .Rating
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(15)
    )

    attraction_summary = attraction_summary.rename(
        columns={
            "count": "Visit Count",
            "mean": "Average Rating"
        }
    )

    st.dataframe(
        attraction_summary,
        use_container_width=True
    )


# ---------------------------------------------------------
# Rating Prediction
# ---------------------------------------------------------
elif page == "Rating Prediction":

    st.header("⭐ Rating Prediction")

    st.write(
        "Select the tourism and attraction characteristics below "
        "to predict the expected attraction rating."
    )

    x = form_inputs()

    if st.button("Predict Rating", type="primary"):

        prediction = rating_model.predict(x)[0]
        prediction = max(1, min(5, prediction))

        st.metric(
            "Predicted Rating",
            f"{prediction:.2f} / 5"
        )


# ---------------------------------------------------------
# Visit Mode Prediction
# ---------------------------------------------------------
elif page == "Visit Mode Prediction":

    st.header("👥 Visit Mode Prediction")

    st.write(
        "Select the tourism and attraction characteristics below "
        "to predict the likely visit mode."
    )

    x = form_inputs()

    if st.button("Predict Visit Mode", type="primary"):

        prediction = mode_model.predict(x)[0]

        st.success(
            f"Predicted Visit Mode: **{prediction}**"
        )

        probs = mode_model.predict_proba(x)[0]

        probability_data = dict(
            zip(mode_model.classes_, probs)
        )

        st.subheader("Prediction Probabilities")

        st.bar_chart(probability_data)


# ---------------------------------------------------------
# Attraction Recommendations
# ---------------------------------------------------------
elif page == "Attraction Recommendations":

    st.header("🗺️ Attraction Recommendations")

    st.write(
        "Explore recommended attractions using popularity-based "
        "recommendations with optional attraction-type and city filters."
    )

    typ = st.selectbox(
        "Optional Attraction Type",
        [""] + sorted(
            master.AttractionType
            .dropna()
            .unique()
            .tolist()
        )
    )

    city = st.selectbox(
        "Optional Attraction City",
        [""] + sorted(
            master.AttractionCity
            .dropna()
            .unique()
            .tolist()
        )
    )

    recommendations = recommendation_table(
        popular,
        typ or None,
        city or None
    )

    st.subheader("Recommended Attractions")

    st.dataframe(
        recommendations[
            [
                "Attraction",
                "AttractionType",
                "AttractionCity",
                "AttractionCountry",
                "recommendation_score",
                "reason"
            ]
        ],
        use_container_width=True
    )


# ---------------------------------------------------------
# Business Insights
# ---------------------------------------------------------
else:

    st.header("📊 Business Insights")

    st.write(
        "Key observations, supporting evidence, and business implications "
        "derived from the tourism analytics and machine learning analysis."
    )

    # Path to the insights JSON file
    insights_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "eda"
        / "business_insights.json"
    )

    # Load insights
    with open(insights_path, "r", encoding="utf-8") as f:
        insights = json.load(f)

    # Display each insight
    for i, insight in enumerate(insights, start=1):

        st.subheader(f"Insight {i}")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 🔎 Observation")

            st.info(
                insight["observation"]
            )

        with col2:

            st.markdown("### 📈 Evidence")

            st.success(
                insight["evidence"]
            )

        st.markdown("### 📌 Business Implication")

        st.write(
            insight["business_implication"]
        )

        st.divider()