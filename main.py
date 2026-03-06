# main.py

import streamlit as st
import pandas as pd
from model_training import train_models


st.title("Census Income Prediction using Machine Learning")

st.write("Upload Census Dataset to train ML models")

# Upload dataset
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    data = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    if st.button("Train Models"):

        results = train_models(data)

        st.subheader("Model Accuracy Results")

        for model, accuracy in results.items():
            st.write(f"{model} Accuracy: {accuracy:.2f}")