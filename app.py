import streamlit as st
import pandas as pd

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# CSS to hide ALL default navigation links
st.markdown("""
<style>
    /* Hide the entire navigation menu */
    section[data-testid="stSidebar"] .st-emotion-cache-1mi2ry5 {
        display: none !important;
    }
    
    /* Hide any navigation elements */
    .st-emotion-cache-1mi2ry5, 
    .st-emotion-cache-1gulkj5,
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Style sidebar buttons */
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
    
    /* Style for home button */
    .home-button .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    
    /* Style for pointing finger */
    .pointing-finger {
        text-align: center;
        font-size: 30px;
        margin: 5px 0 15px 0;
        color: #4CAF50;
        animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(10px); }
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# SIDEBAR NAVIGATION
# ===========================================
with st.sidebar:
    st.title("📌 Navigation")
    st.markdown("---")
    
    # Sidebar navigation buttons with correct filenames
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

st.title("🏢 Employee Attrition Dashboard")

st.markdown("""
This app explores key factors impacting employee attrition data using **data analysis, visualisation, and a machine learning model**.
Navigate using the buttons below:
""")
# Finger pointing down icon
st.markdown('<div class="pointing-finger">👇</div>', unsafe_allow_html=True)

# Home button with pin icon
col_home, _ = st.columns([1, 5])
with col_home:
    st.markdown('<div class="home-button">', unsafe_allow_html=True)
    if st.button("📌 Home", key="main_home", use_container_width=True):
        st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)


# Three main page buttons with correct filenames
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Overview", key="main_overview", use_container_width=True):
        st.switch_page("pages/1_overview.py")
    st.caption("Dataset overview and basic statistics")

with col2:
    if st.button("🧪 Data Visualisation", key="main_viz", use_container_width=True):
        st.switch_page("pages/2_visuals.py")
    st.caption("Explore visual insights and patterns")

with col3:
    if st.button("🤖 ML Predictor", key="main_predictor", use_container_width=True):
        st.switch_page("pages/3_ml_model.py")
    st.caption("Predict employee attrition risk")


# Load and display data preview
try:
    df = pd.read_csv("data/processed_attrition_dataset.csv")
    
    st.subheader("Quick Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)
        
except FileNotFoundError:
    st.error("⚠️ Dataset not found. Please ensure 'data/processed_attrition_dataset.csv' exists.")
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")

st.markdown("---")
st.caption("Employee Attrition Dashboard - Powered by Streamlit")