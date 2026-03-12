import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Data Overview - Employee Attrition", layout="wide")

# CSS to hide default navigation links
st.markdown("""
<style>
    /* Hide the default page navigation links */
    section[data-testid="stSidebar"] .st-emotion-cache-1mi2ry5 {
        display: none !important;
    }
    
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Style sidebar buttons */
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# SIDEBAR NAVIGATION
# ===========================================
with st.sidebar:
    st.title("📌 Navigation")
    st.markdown("---")
    
    # Home button
    if st.button("🏠 Home", key="sidebar_home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Page navigation buttons
    if st.button("📊 Overview", key="sidebar_overview"):
        st.switch_page("pages/1_overview.py")
    
    if st.button("🧪 Data Visualisation", key="sidebar_viz"):
        st.switch_page("pages/2_visuals.py")
    
    if st.button("🤖 ML Predictor", key="sidebar_predictor"):
        st.switch_page("pages/3_ml_model.py")
    
    st.markdown("---")
    st.caption("Employee Attrition Dashboard v1.0")

# ===========================================
# MAIN CONTENT
# ===========================================
st.title("📊 Data Overview")

# Check if file exists
file_path = "data/processed_attrition_dataset.csv"

if not os.path.exists(file_path):
    st.error(f"❌ Dataset not found at {file_path}. Please ensure the file exists.")
    st.stop()

try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

st.markdown("""
This page provides an overview of the employee attrition dataset, including key statistics
and a summary of the data used in the analysis.
""")

# KPI metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Employees", f"{len(df):,}")

with col2:
    if 'Age' in df.columns:
        st.metric("Median Age", f"{df['Age'].median():.1f}")
    else:
        st.metric("Median Age", "N/A")

with col3:
    if 'Attrition' in df.columns:
        # Handle different possible formats (Yes/No, 1/0, True/False)
        attrition_col = df['Attrition'].astype(str).str.lower()
        attrition_pct = (attrition_col.isin(['yes', '1', 'true', 'y'])).mean() * 100
        st.metric("Attrition Rate", f"{attrition_pct:.1f}%")
    else:
        st.metric("Attrition Rate", "N/A")

st.divider()

# Dataset preview
st.subheader("Dataset Preview (First 10 Rows)")
st.dataframe(df.head(10), use_container_width=True)

# Dataset dimensions
col1, col2 = st.columns(2)
with col1:
    st.info(f"📏 **Rows:** {df.shape[0]:,}")
with col2:
    st.info(f"📐 **Columns:** {df.shape[1]:,}")

st.divider()

# Missing values check
st.subheader("Missing Values Check")
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing Values': df.isnull().sum().values,
    'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
})
st.dataframe(missing_df, use_container_width=True)

st.divider()

# Data types
st.subheader("Data Types")
dtype_df = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values
})
st.dataframe(dtype_df, use_container_width=True)

st.divider()

# Basic statistics
st.subheader("Basic Statistics (Numerical Columns)")
st.dataframe(df.describe(), use_container_width=True)

# Footer
st.divider()
st.caption("Employee Attrition Dashboard - Overview Page")