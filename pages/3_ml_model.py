import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ===========================================
# Define the custom class for loading the model
# ===========================================
class PipelineClusterNumeric:
    """
    Custom pipeline class that matches the one used during training
    """
    def __init__(self, n_clusters=4, n_pca_components=5, random_state=42):
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components, random_state=random_state)),
            ('cluster', KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10))
        ])
        self.named_steps = self.pipeline.named_steps
    
    def fit(self, X):
        self.pipeline.fit(X)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def fit_predict(self, X):
        return self.pipeline.fit_predict(X)

st.set_page_config(page_title="ML Predictor - Employee Attrition", layout="wide")

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
    
    /* Style for prediction results */
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .medium-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .good-risk {
        background-color: #d1e7dd;
        color: #0a3622;
        border: 1px solid #a3cfbb;
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
st.title("🤖 Employee Attrition Risk Predictor")

st.markdown("""
This page uses a machine learning model to predict employee attrition risk based on key features.
Enter the employee's information below to get a cluster assignment and risk assessment.
""")

st.divider()

# -------------------------
# 1. Load the saved model
# -------------------------
# Using path relative to this file (pages/3_ml_model.py)
MODEL_PATH = Path(__file__).parent.parent / "models" / "employee_attrition_cluster3.pkl"

# Debug info in expander
with st.expander("🔧 Debug Info", expanded=False):
    st.write(f"Looking for model at: {MODEL_PATH}")
    st.write(f"File exists: {os.path.exists(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Train the model first!")
    st.info("""
    **Troubleshooting:**
    - Make sure you've trained the model first
    - Check that the model file exists in the 'models' folder
    - Expected path: `models/employee_attrition_cluster3.pkl`
    """)
    st.stop()

try:
    model_package = joblib.load(MODEL_PATH)
    pipeline = model_package["pipeline"]
    top_features = model_package["top_features"]
    cluster_risk_mapping = model_package.get("cluster_risk_mapping", {})
    attrition_rates = model_package.get("attrition_rates", {})
    
    st.success("✅ Model loaded successfully!")
    
    # Show model info in expander
    with st.expander("📊 Model Information", expanded=False):
        st.write(f"**Number of Clusters:** {model_package.get('n_clusters', 4)}")
        st.write(f"**Top Features:** {', '.join(top_features)}")
        if attrition_rates:
            st.write("**Attrition Rates by Cluster:**")
            for cluster, rate in attrition_rates.items():
                st.write(f"  - Cluster {cluster}: {rate:.1%}")
        
        if cluster_risk_mapping:
            st.write("**Risk Mapping (from training):**")
            for cluster, risk in cluster_risk_mapping.items():
                st.write(f"  - Cluster {cluster}: {risk}")
                
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.divider()

# -------------------------
# 2. User input for features
# -------------------------
st.subheader("📝 Enter Employee Features")

# Create two columns for better layout
col1, col2 = st.columns(2)

feature_inputs = {}
for i, feature in enumerate(top_features[:6]):  # Take first 6 features
    # Alternate between columns
    with col1 if i % 2 == 0 else col2:
        # Set sensible ranges based on typical HR data
        if "Age" in feature:
            default_val = 35
            min_val = 18
            max_val = 70
        elif "MonthlyIncome" in feature or "Income" in feature:
            default_val = 5000
            min_val = 1000
            max_val = 20000
        elif "Experience" in feature or "Years" in feature:
            default_val = 5
            min_val = 0
            max_val = 40
        elif "Score" in feature or "Rating" in feature:
            default_val = 3
            min_val = 1
            max_val = 5
        else:
            default_val = 0
            min_val = 0
            max_val = 100
        
        feature_inputs[feature] = st.number_input(
            label=f"**{feature}**",
            value=int(default_val),
            min_value=int(min_val),
            max_value=int(max_val),
            help=f"Enter value for {feature}"
        )

st.divider()

# Convert inputs to DataFrame
input_df = pd.DataFrame([feature_inputs])

# Show input summary
with st.expander("📋 View Input Summary", expanded=False):
    st.dataframe(input_df, use_container_width=True)

# -------------------------
# 3. Predict cluster on button click
# -------------------------
col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
with col_pred2:
    predict_button = st.button("🔮 Predict Cluster & Risk", use_container_width=True)

if predict_button:
    try:
        # Make prediction
        cluster_label = pipeline.predict(input_df)[0]
        
        # Get risk label from the model's trained mapping
        # This ensures consistency with how the model was trained
        if cluster_risk_mapping and int(cluster_label) in cluster_risk_mapping:
            risk_label = cluster_risk_mapping[int(cluster_label)]
        else:
            # Fallback if mapping not available
            if cluster_label == 0:
                risk_label = "🔴 HIGH RISK"
            elif cluster_label == 1:
                risk_label = "🟡 MEDIUM RISK"
            elif cluster_label == 2:
                risk_label = "🟢 LOW RISK"
            elif cluster_label == 3:
                risk_label = "💚 GOOD"
            else:
                risk_label = f"Cluster {cluster_label}"
        
        # Display results
        st.subheader("🎯 Prediction Result")
        
        # Create result columns
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Predicted Cluster", f"Cluster {cluster_label}")
        
        with res_col2:
            # Style based on risk level
            if "HIGH RISK" in risk_label:
                st.markdown('<div class="prediction-box high-risk">', unsafe_allow_html=True)
            elif "MEDIUM RISK" in risk_label:
                st.markdown('<div class="prediction-box medium-risk">', unsafe_allow_html=True)
            elif "LOW RISK" in risk_label:
                st.markdown('<div class="prediction-box low-risk">', unsafe_allow_html=True)
            else:  # GOOD
                st.markdown('<div class="prediction-box good-risk">', unsafe_allow_html=True)
            
            st.metric("Risk Level", risk_label)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show cluster interpretation
        st.subheader("📊 Cluster Interpretation")
        
        # Get attrition rate for this cluster
        cluster_rate = attrition_rates.get(int(cluster_label), "N/A")
        rate_text = f" ({cluster_rate:.1%} attrition rate)" if isinstance(cluster_rate, (int, float)) else ""
        
        # Create interpretation based on risk label from trained mapping
        if "HIGH RISK" in risk_label:
            interpretation = f"🔴 **HIGH RISK CLUSTER**{rate_text} - These employees have the highest probability of leaving. Immediate intervention recommended: salary review, career development, engagement initiatives."
        elif "MEDIUM RISK" in risk_label:
            interpretation = f"🟡 **MEDIUM RISK CLUSTER**{rate_text} - Moderate attrition probability. Monitor these employees and consider proactive retention strategies."
        elif "LOW RISK" in risk_label:
            interpretation = f"🟢 **LOW RISK CLUSTER**{rate_text} - Stable employees with low attrition probability. Continue current engagement practices."
        else:  # GOOD
            interpretation = f"💚 **GOOD STANDING**{rate_text} - Most stable employees with lowest attrition probability. Continue current practices."
        
        st.info(interpretation)
        
        # Show all clusters comparison
        if attrition_rates:
            st.subheader("📈 All Clusters Comparison")
            
            # Create comparison dataframe
            comparison = []
            for cluster, rate in attrition_rates.items():
                # Get risk label from mapping
                risk_display = cluster_risk_mapping.get(int(cluster), f"Cluster {cluster}")
                comparison.append({
                    "Cluster": f"Cluster {cluster}",
                    "Attrition Rate": f"{rate:.1%}",
                    "Risk Level": risk_display.split('(')[0].strip() if '(' in risk_display else risk_display,
                    "Current": "✓" if cluster == cluster_label else ""
                })
            
            # Sort by attrition rate (highest first)
            comparison_df = pd.DataFrame(comparison)
            comparison_df = comparison_df.sort_values("Attrition Rate", ascending=False)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Highlight current cluster
            if "HIGH RISK" in risk_label:
                st.error(f"📍 Current employee is in the HIGHEST risk cluster")
            elif "GOOD" in risk_label:
                st.success(f"📍 Current employee is in the BEST standing (GOOD)")
            else:
                st.info(f"📍 Current employee risk level: {risk_label.split('(')[0].strip() if '(' in risk_label else risk_label}")
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.exception(e)

# Footer
st.divider()
st.caption("Employee Attrition Dashboard - ML Predictor Page")
