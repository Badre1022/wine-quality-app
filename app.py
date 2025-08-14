# app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/winequality-red.csv")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Sidebar navigation
st.sidebar.title("Wine Quality Prediction App")
menu = st.sidebar.radio("Navigation", ["Data Exploration", "Visualization", "Prediction", "Model Performance"])

# --- Data Exploration ---
if menu == "Data Exploration":
    st.title("Wine Quality Dataset Overview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    st.write(df.describe())
    if st.checkbox("Show missing values"):
        st.write(df.isnull().sum())

# --- Visualization ---
elif menu == "Visualization":
    st.title("Data Visualization")
    fig = px.histogram(df, x="quality", title="Quality Distribution")
    st.plotly_chart(fig)
    fig_corr, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

# --- Prediction ---
elif menu == "Prediction":
    st.title("Predict Wine Quality")
    features = {}
    for col in df.columns[:-1]:  # exclude quality
        features[col] = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]
    st.write("Prediction:", "Good" if prediction == 1 else "Not Good")
    st.write("Confidence:", round(proba * 100, 2), "%")

# --- Model Performance ---
elif menu == "Model Performance":
    st.title("Model Performance Metrics")
    st.write("This section will show accuracy, confusion matrix, and other metrics.")
