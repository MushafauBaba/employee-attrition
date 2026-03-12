import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Add this line
import os

st.set_page_config(page_title="Data Visualisation - Employee Attrition", layout="wide")

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
st.title("🧪 Data Visualisations")

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
This page provides visual insights into the employee attrition data, including attrition distribution
and correlation analysis between key features.
""")

st.divider()

# ----------------------------
# Attrition Percentage of Leavers
# ----------------------------
st.subheader("📊 Percentage of Employee Attrition")

st.markdown("""
**Interpretation:**  
The below charts shows us that 16% of the 1470 employees recorded during the analysis decided to leave for a particular reason.  
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Check if Attrition column exists
if 'Attrition' in df.columns:
    Risk_counts = df['Attrition'].value_counts()
    
    colors = ['#2ecc71', '#FF0000']
    
    # Pie chart
    axes[0].pie(Risk_counts, labels=['No', 'Yes'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('Attrition Risk Distribution', fontsize=12)
    
    # Bar chart
    axes[1].bar(['No', 'Yes'], Risk_counts, color=colors)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Attrition Count', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(Risk_counts):
        axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
else:
    axes[0].text(0.5, 0.5, 'Attrition column not found', ha='center')
    axes[1].text(0.5, 0.5, 'Attrition column not found', ha='center')

plt.tight_layout()
st.pyplot(fig)

# Display summary stats
if 'Attrition' in df.columns:
    attrition_count = df['Attrition'].astype(str).str.lower().isin(['yes', '1', 'true', 'y']).sum()
    total_count = len(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", f"{total_count:,}")
    with col2:
        st.metric("Attrition Count", f"{attrition_count:,}")
    with col3:
        st.metric("Attrition Rate", f"{(attrition_count/total_count*100):.1f}%")

st.divider()

# ----------------------------
# Feature Correlation Matrix
# ----------------------------
st.subheader("🔗 Correlation Analysis of Features")

st.markdown("""
**Interpretation:**  
During the EDA process we wanted to see if there was any correlation between the following key features:
- Performance Rating and Number of Trainings
- Salary Hike Percent and Job Satisfaction

Based on the below heatmap the analysis shows that there is no correlation or relationship existing between Performance rating and the number of training in the last year.
There is also no correlation between Salary hike and Job satisfaction.

However, there is a strong **0.77 correlation coefficient** between Salary Hike percent and Performance Rating,
which means there is strong linear relationship between salary hike and performance rating. Generally, higher salary hike results to higher performance rating and vice versa.
""")

# Select only numeric columns for correlation
num_cols = df.select_dtypes(include="number").columns

if len(num_cols) > 1:
    correlation_matrix = df[num_cols].corr()
    
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, linewidths=1, ax=ax2, fmt='.2f',
                cbar_kws={"shrink": 0.8})
    ax2.set_title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    
    # Show strongest correlations
    st.subheader("🔍 Top Correlations")
    
    # Get upper triangle of correlation matrix
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Stack and sort
    strong_corr = upper_tri.unstack().dropna().sort_values(ascending=False)
    
    if len(strong_corr) > 0:
        corr_df = strong_corr.head(10).reset_index()
        corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
        st.dataframe(corr_df, use_container_width=True)
else:
    st.warning("Not enough numeric columns for correlation analysis.")

st.divider()

# ----------------------------
# Additional Insights
# ----------------------------
st.subheader("💡 Key Insights Summary")

col_ins1, col_ins2 = st.columns(2)

with col_ins1:
    st.info("""
    **📈 Attrition Insights**
    - 16% of employees left the company
    - Majority of employees (84%) stayed
    """)

with col_ins2:
    st.success("""
    **🔗 Correlation Insights**
    - Strong correlation (0.77) between salary hike and performance
    - No correlation between performance rating and training count
    - No correlation between salary hike and job satisfaction
    """)

# Footer
st.divider()
st.caption("Employee Attrition Dashboard - Visualisations Page")