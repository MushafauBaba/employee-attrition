import streamlit as st
import pandas as pd

st.title("📊 Data Overview")

df = pd.read_csv("data/processed_attrition_dataset.csv")

st.markdown("""
This page provides an overview of the employee attrition dataset, including key statistics
and a summary of the data used in the analysis.
""")

# KPI metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", f"{len(df):,}")
col2.metric("Median Age", f"{df['Age'].median():,.2f}")

attrition_pct = (df["Attrition"].astype(str).str.lower() == "yes").mean() * 100
col3.metric("Attrition (%)", f"{attrition_pct:.1f}%")

st.divider()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Missing Values Check")
st.write(df.isnull().sum())
