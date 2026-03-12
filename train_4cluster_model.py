# train_3cluster_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class PipelineClusterNumeric:
    """
    Custom pipeline class that mimics your original structure
    """
    def __init__(self, n_clusters=3, n_pca_components=5, random_state=42):  # Changed to 3
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

def train_and_save_cluster_model(
    X: pd.DataFrame,
    save_path: str = "models/employee_attrition_cluster3.pkl",
    top_features: list = ["YearsAtCompany", "TotalWorkingYears", "MonthlyIncome",
                          "YearsWithCurrManager", "YearsInCurrentRole"]
):
    """
    Train KMeans clustering with 3 clusters on top numeric features, 
    interpret clusters using Attrition, and save pipeline + metadata for Streamlit.
    """

    print("="*60)
    print("TRAINING 3-CLUSTER MODEL")
    print("="*60)
    
    # -------------------------
    # 1. Prepare features
    # -------------------------
    X_features = X[top_features].copy()
    print(f"📊 Training on {len(X_features)} samples with features: {top_features}")
    
    # -------------------------
    # 2. Train pipeline with 3 clusters
    # -------------------------
    pipeline_cluster = PipelineClusterNumeric(n_clusters=3, n_pca_components=min(len(top_features), 5))
    pipeline_cluster.fit(X_features)
    
    # -------------------------
    # 3. Assign clusters
    # -------------------------
    X = X.copy()
    X["Cluster"] = pipeline_cluster.predict(X_features)
    
    # Check cluster distribution
    cluster_dist = X["Cluster"].value_counts().sort_index()
    print(f"\n📊 Cluster Distribution:")
    for cluster in range(3):  # Changed to 3
        count = cluster_dist.get(cluster, 0)
        print(f"  Cluster {cluster}: {count} employees ({count/len(X)*100:.1f}%)")
    
    # -------------------------
    # 4. Interpret clusters with Attrition
    # -------------------------
    if "Attrition" in X.columns:
        # Convert Attrition to numeric
        X["Attrition_num"] = X["Attrition"].map({"Yes": 1, "No": 0})
        
        # Calculate attrition rate per cluster
        attrition_rates = X.groupby("Cluster")["Attrition_num"].mean()
        
        print(f"\n📊 Attrition Rates by Cluster (before sorting):")
        for cluster, rate in attrition_rates.items():
            leavers = int(rate * X[X['Cluster']==cluster].shape[0])
            total = X[X['Cluster']==cluster].shape[0]
            print(f"  Cluster {cluster}: {rate:.1%} ({leavers}/{total} leavers)")
        
        # Sort clusters by attrition rate (highest first) for risk mapping
        sorted_clusters = attrition_rates.sort_values(ascending=False).index.tolist()
        
        print(f"\n📊 Clusters sorted by attrition rate (highest to lowest):")
        for i, cluster in enumerate(sorted_clusters):
            print(f"  Rank {i+1}: Cluster {cluster} ({attrition_rates[cluster]:.1%})")
        
        # Map clusters to risk levels based on attrition rates (highest rate = HIGHEST RISK)
        # Risk levels in order: HIGHEST risk → MEDIUM risk → LOW risk
        risk_labels = ["🔴 HIGH RISK", "🟡 MEDIUM RISK", "🟢 LOW RISK"]
        
        cluster_risk_mapping = {}
        print(f"\n🏷 Risk Mapping (highest attrition = HIGHEST RISK):")
        
        for i, cluster in enumerate(sorted_clusters):
            rate = attrition_rates[cluster]
            
            if i == 0:  # Highest attrition rate
                risk_label = f"{risk_labels[0]} ({rate:.1%})"
                print(f"  Cluster {cluster}: {risk_label} ← HIGHEST attrition = HIGHEST RISK")
            elif i == 1:  # Second highest
                risk_label = f"{risk_labels[1]} ({rate:.1%})"
                print(f"  Cluster {cluster}: {risk_label}")
            else:  # Lowest attrition rate
                risk_label = f"{risk_labels[2]} ({rate:.1%})"
                print(f"  Cluster {cluster}: {risk_label} ← LOWEST attrition = LOW RISK")
            
            cluster_risk_mapping[int(cluster)] = risk_label
            
    else:
        attrition_rates = pd.Series(dtype=float)
        cluster_risk_mapping = {i: f"Cluster {i}" for i in range(3)}  # Changed to 3
    
    # -------------------------
    # 5. Package model - PRODUCTION READY FIX
    # -------------------------
    # Extract the underlying sklearn pipeline (no custom class dependency)
    simple_sklearn_pipeline = pipeline_cluster.pipeline
    
    # Create model package with plain sklearn pipeline
    model_package = {
        "pipeline": simple_sklearn_pipeline,  # ✅ Plain sklearn Pipeline - NO custom class needed!
        "top_features": top_features,
        "cluster_risk_mapping": cluster_risk_mapping,
        "attrition_rates": attrition_rates.to_dict() if not attrition_rates.empty else {},
        "n_clusters": 3,  # Changed to 3
        "cluster_distribution": X["Cluster"].value_counts().to_dict(),
        "risk_mapping_order": "HIGHEST attrition = HIGHEST RISK"
    }

    # -------------------------
    # 6. Save to disk (overwrite existing)
    # -------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Remove old file if exists
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\n🗑️ Removed old model: {save_path}")
    
    joblib.dump(model_package, save_path)
    
    print(f"\n✅ New 3-cluster model saved to: {save_path}")
    print(f"✅ Model saved in production-ready format (no custom class dependencies)")
    
    # Verify save
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path) / 1024  # KB
        print(f"✅ File saved successfully ({file_size:.1f} KB)")
    else:
        print(f"❌ Error: File was not saved!")
    
    return model_package

# ===========================================
# RUN THE TRAINING
# ===========================================
if __name__ == "__main__":
    print("🚀 Starting 3-Cluster Model Training...")
    
    # Load CSV
    data_path = "data/processed_attrition_dataset.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
    else:
        X = pd.read_csv(data_path)
        print(f"✅ Loaded data: {X.shape[0]} rows, {X.shape[1]} columns")
        
        # Train model and save (overwrites existing)
        model_package = train_and_save_cluster_model(X)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print("\nRisk Mapping Summary:")
        for cluster, risk in model_package["cluster_risk_mapping"].items():
            print(f"  Cluster {cluster}: {risk}")
        print("\n" + "="*60)
        print("📦 PRODUCTION-READY MODEL SAVED")
        print("="*60)
        print("✅ 3-Cluster Model with:")
        print("✅ 🔴 HIGH RISK (highest attrition)")
        print("✅ 🟡 MEDIUM RISK")
        print("✅ 🟢 LOW RISK (lowest attrition)")
        print("="*60)