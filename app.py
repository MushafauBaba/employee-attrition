import streamlit as st
import pandas as pd

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("🏥 Employee Attrition Dashboard")

st.markdown("""
This app explores key factors impacting employee attrition data using **data analysis, visualisation, and a machine learning model**.

Use the **sidebar** to navigate:
- 📊 Overview
- 🧪 Data Visualisation
- 🤖 ML Predictor
""")

df = pd.read_csv("data/processed_attrition_dataset.csv")

st.subheader("Quick Preview")
st.markdown("Here's a quick look at the first 10 rows of the dataset, along with some basic statistics and information about the data types.")
st.dataframe(df.head(10), use_container_width=True)
st.markdown("### Basic Statistics")
st.dataframe(df.describe(), use_container_width=True)
st.markdown("### Data Information")
st.dataframe(df.info(), use_container_width=True)

