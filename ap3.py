# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------
# 1. Load the saved model
# -------------------------
MODEL_PATH = "models/employee_attrition_cluster3.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Train the model first!")
    st.stop()

model_package = joblib.load(MODEL_PATH)
pipeline = model_package["pipeline"]
top_features = model_package["top_features"]
cluster_risk_mapping = model_package.get("cluster_risk_mapping", {})

st.title("Employee Attrition Risk Clustering")
st.write("Enter the employee's feature values to estimate cluster and attrition risk.")

# -------------------------
# 2. User input for the 5 numeric features
# -------------------------
feature_inputs = {}
for feature in top_features:
    # Set a sensible range using +/- 3 std dev if known, otherwise just default min/max
    default_val = 0
    min_val = 0
    max_val = 100  # arbitrary max, can adjust
    feature_inputs[feature] = st.number_input(
        label=f"{feature}",
        value=int(default_val),
        min_value=int(min_val),
        max_value=int(max_val)
    )

# Convert inputs to DataFrame
input_df = pd.DataFrame([feature_inputs])

# -------------------------
# 3. Predict cluster on button click
# -------------------------
if st.button("Predict Cluster & Risk"):
    cluster_label = pipeline.predict(input_df)[0]
    risk_label = cluster_risk_mapping.get(int(cluster_label), "Unknown Risk")

    st.subheader("Prediction Result")
    st.write(f"**Cluster:** {cluster_label}")
    st.write(f"**Risk Level:** {risk_label}")

    if "attrition_rates" in model_package and model_package["attrition_rates"]:
        st.write("📊 Cluster attrition rates (from training data):")
        st.json(model_package["attrition_rates"])
