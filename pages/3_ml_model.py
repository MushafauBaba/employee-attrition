import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# ===========================================
# Define the custom class for loading the model
# ===========================================
class PipelineClusterNumeric:
    """
    Custom pipeline class that matches the one used during training
    """
    def __init__(self, n_clusters=3, n_pca_components=5, random_state=42):
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
MODEL_PATH = Path(__file__).parent.parent / "models" / "employee_attrition_cluster3.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Train the model first!")
    st.stop()

try:
    model_package = joblib.load(MODEL_PATH)
    pipeline = model_package["pipeline"]
    top_features = model_package["top_features"]
    cluster_risk_mapping = model_package.get("cluster_risk_mapping", {})
    attrition_rates = model_package.get("attrition_rates", {})
    
    st.success("✅ Model loaded successfully!")
                
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
for i, feature in enumerate(top_features[:6]):
    with col1 if i % 2 == 0 else col2:
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
        cluster_label = pipeline.predict(input_df)[0]
        
        if cluster_risk_mapping and int(cluster_label) in cluster_risk_mapping:
            risk_label = cluster_risk_mapping[int(cluster_label)]
        else:
            if cluster_label == 0:
                risk_label = "🔴 HIGH RISK"
            elif cluster_label == 1:
                risk_label = "🟡 MEDIUM RISK"
            elif cluster_label == 2:
                risk_label = "🟢 LOW RISK"
            else:
                risk_label = f"Cluster {cluster_label}"
        
        st.subheader("🎯 Prediction Result")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Predicted Cluster", f"Cluster {cluster_label}")
        
        with res_col2:
            if "HIGH RISK" in risk_label:
                st.markdown('<div class="prediction-box high-risk">', unsafe_allow_html=True)
            elif "MEDIUM RISK" in risk_label:
                st.markdown('<div class="prediction-box medium-risk">', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box low-risk">', unsafe_allow_html=True)
            
            st.metric("Risk Level", risk_label)
            st.markdown('</div>', unsafe_allow_html=True)
        
        cluster_rate = attrition_rates.get(int(cluster_label), "N/A")
        rate_text = f" ({cluster_rate:.1%} attrition rate)" if isinstance(cluster_rate, (int, float)) else ""
        
        if "HIGH RISK" in risk_label:
            interpretation = f"🔴 **HIGH RISK CLUSTER**{rate_text} - These employees have the highest probability of leaving."
        elif "MEDIUM RISK" in risk_label:
            interpretation = f"🟡 **MEDIUM RISK CLUSTER**{rate_text} - Moderate attrition probability."
        else:
            interpretation = f"🟢 **LOW RISK CLUSTER**{rate_text} - Stable employees with low attrition probability."
        
        st.info(interpretation)
        
        if attrition_rates:
            st.subheader("📈 All Clusters Comparison")
            
            comparison = []
            for cluster, rate in attrition_rates.items():
                risk_display = cluster_risk_mapping.get(int(cluster), f"Cluster {cluster}")
                comparison.append({
                    "Cluster": f"Cluster {cluster}",
                    "Attrition Rate": f"{rate:.1%}",
                    "Risk Level": risk_display.split('(')[0].strip() if '(' in risk_display else risk_display,
                    "Current": "✓" if cluster == cluster_label else ""
                })
            
            comparison_df = pd.DataFrame(comparison)
            comparison_df = comparison_df.sort_values("Attrition Rate", ascending=False)
            st.dataframe(comparison_df, use_container_width=True)
            
            if "HIGH RISK" in risk_label:
                st.error(f"📍 Current employee is in the HIGHEST risk cluster")
            elif "LOW RISK" in risk_label:
                st.success(f"📍 Current employee is in the LOWEST risk cluster")
            else:
                st.info(f"📍 Current employee is in MEDIUM risk cluster")
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.exception(e)

st.divider()

# ===========================================
# CLUSTER PROFILING SECTION
# ===========================================
st.header("📊 Employee Cluster Profiles")

try:
    full_data = pd.read_csv("data/processed_attrition_dataset.csv")
    
    # Add cluster labels from the model
    X_features = full_data[top_features].copy()
    full_data["Cluster"] = pipeline.predict(X_features)
    
    # Calculate average values for each cluster
    if "Attrition" in full_data.columns:
        full_data["Attrition_Num"] = full_data["Attrition"].map({"Yes": 1, "No": 0})
    
    # Create profile dataframe
    profile_df = full_data.groupby("Cluster")[top_features].mean().round(2)
    
    # Add attrition rate
    if "Attrition_Num" in full_data.columns:
        attrition_by_cluster = full_data.groupby("Cluster")["Attrition_Num"].mean() * 100
        profile_df["Attrition Rate (%)"] = (attrition_by_cluster).round(1)
    
    # Add risk level from model mapping
    risk_levels = []
    for cluster in profile_df.index:
        if cluster_risk_mapping and int(cluster) in cluster_risk_mapping:
            risk = cluster_risk_mapping[int(cluster)]
            risk_levels.append(risk.split('(')[0].strip())
        else:
            risk_levels.append("Unknown")
    profile_df["Risk Level"] = risk_levels
    
    # Reorder columns
    cols = ["Risk Level", "Attrition Rate (%)"] + [c for c in profile_df.columns if c not in ["Risk Level", "Attrition Rate (%)"]]
    profile_df = profile_df[cols]
    
    # Display profile table
    st.subheader("📋 Cluster Average Values")
    st.dataframe(profile_df, use_container_width=True)
    
    # Create radar chart for cluster comparison (like weather clusters)
    st.subheader("🔄 Cluster Characteristics Radar Chart")
    
    # Normalize data for radar chart
    from sklearn.preprocessing import MinMaxScaler
    
    # Select numeric columns for radar
    radar_cols = [c for c in top_features if c in profile_df.columns]
    radar_df = profile_df[radar_cols].copy()
    
    # Normalize
    scaler = MinMaxScaler()
    radar_normalized = pd.DataFrame(
        scaler.fit_transform(radar_df),
        columns=radar_df.columns,
        index=radar_df.index
    )
    
    # Create radar chart
    from math import pi
    
    categories = radar_cols
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
    
    colors = ['#f8d7da', '#fff3cd', '#d4edda']  # red, yellow, green
    
    for idx, cluster in enumerate(sorted(profile_df.index)):
        ax = axes[idx]
        values = radar_normalized.loc[cluster].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        
        risk = profile_df.loc[cluster, "Risk Level"]
        att_rate = profile_df.loc[cluster, "Attrition Rate (%)"]
        ax.set_title(f'Cluster {cluster}: {risk}\n({att_rate:.1f}% attrition)', size=10, pad=15)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Individual cluster interpretations
    st.subheader("🔍 Cluster Deep Dive")
    
    col_int1, col_int2, col_int3 = st.columns(3)
    cols = [col_int1, col_int2, col_int3]
    
    for idx, cluster in enumerate(sorted(profile_df.index)):
        with cols[idx]:
            risk = profile_df.loc[cluster, "Risk Level"]
            att_rate = profile_df.loc[cluster, "Attrition Rate (%)"]
            
            if "HIGH RISK" in risk:
                st.markdown(f"### 🔴 Cluster {cluster}")
            elif "MEDIUM RISK" in risk:
                st.markdown(f"### 🟡 Cluster {cluster}")
            else:
                st.markdown(f"### 🟢 Cluster {cluster}")
            
            st.markdown(f"**Risk Level:** {risk}")
            st.markdown(f"**Attrition Rate:** {att_rate:.1f}%")
            st.markdown("**Average Values:**")
            
            for feat in top_features:
                val = profile_df.loc[cluster, feat]
                if "Income" in feat or "Salary" in feat:
                    st.markdown(f"- {feat}: **${val:,.0f}**")
                elif "Years" in feat or "Age" in feat:
                    st.markdown(f"- {feat}: **{val:.1f} years**")
                else:
                    st.markdown(f"- {feat}: **{val:.2f}**")
    
    # Show cluster sizes
    st.subheader("📊 Cluster Distribution")
    cluster_sizes = full_data["Cluster"].value_counts().sort_index()
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors_bar = ['#f8d7da', '#fff3cd', '#d4edda']
    bars = ax2.bar(cluster_sizes.index, cluster_sizes.values, color=colors_bar)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Employees')
    ax2.set_title('Employee Distribution Across Clusters')
    
    for bar, cluster in zip(bars, cluster_sizes.index):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'n={height}\n({height/len(full_data)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig2)
    
    # How the current input compares
    if predict_button:
        st.subheader("📍 Your Employee vs Cluster Averages")
        
        comparison_data = []
        for feat in top_features:
            if feat in feature_inputs:
                user_val = feature_inputs[feat]
                cluster_val = profile_df.loc[cluster_label, feat]
                
                if "Income" in feat or "Salary" in feat:
                    user_display = f"${user_val:,.0f}"
                    cluster_display = f"${cluster_val:,.0f}"
                elif "Years" in feat or "Age" in feat:
                    user_display = f"{user_val:.1f} years"
                    cluster_display = f"{cluster_val:.1f} years"
                else:
                    user_display = f"{user_val:.2f}"
                    cluster_display = f"{cluster_val:.2f}"
                
                comparison_data.append({
                    "Feature": feat,
                    "Your Value": user_display,
                    f"Cluster {cluster_label} Average": cluster_display
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Show how they compare
        st.markdown("**How you compare to this cluster:**")
        
        for feat in top_features[:3]:  # Show top 3 differences
            if feat in feature_inputs:
                user_val = feature_inputs[feat]
                cluster_val = profile_df.loc[cluster_label, feat]
                
                if user_val > cluster_val * 1.2:
                    st.info(f"✓ Your {feat} is **higher** than cluster average")
                elif user_val < cluster_val * 0.8:
                    st.info(f"⚠️ Your {feat} is **lower** than cluster average")
                else:
                    st.info(f"→ Your {feat} is **similar** to cluster average")
                    
except Exception as e:
    st.warning(f"Could not generate cluster profiles: {e}")

# Footer
st.divider()
st.caption("Employee Attrition Dashboard - ML Predictor Page")