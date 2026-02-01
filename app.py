import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.feature_engineering import add_features

price_model = joblib.load("models/diamond_price_model.pkl")
cluster_model = joblib.load("models/diamond_cluster_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("💎 Diamond Price Prediction & Market Segmentation")

carat = st.number_input("Carat", 0.2, 5.0, 1.0)
x = st.number_input("Length (x)", 3.0, 10.0, 5.0)
y = st.number_input("Width (y)", 3.0, 10.0, 5.0)
z = st.number_input("Depth (z)", 2.0, 8.0, 3.0)

cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

input_df = pd.DataFrame([{
    "carat": carat,
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": (z / ((x+y)/2)) * 100,
    "table": 57,
    "x": x,
    "y": y,
    "z": z
}])

input_df = add_features(input_df)

if st.button("Predict Price"):
    price = price_model.predict(input_df)[0]
    st.success(f"Predicted Price: ₹ {int(price):,}")

if st.button("Predict Market Segment"):
    numeric = input_df.select_dtypes(include=np.number)
    scaled = scaler.transform(numeric)
    cluster = cluster_model.predict(scaled)[0]

    names = {
        0: "Affordable Small Diamonds",
        1: "Mid-range Balanced Diamonds",
        2: "Premium Heavy Diamonds"
    }

    st.info(f"Market Segment: {names[cluster]}")
