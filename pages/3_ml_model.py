import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

st.title("🤖 Machine Learning: Claim Prediction (Linear Regression)")

st.markdown("""
This page demonstrates a simple **Linear Regression** model to predict **insurance claim amounts**
using a small set of numeric and binary patient features.  
The goal is to show a clear, beginner-friendly machine learning workflow.
""")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/insurance_data_clean.csv")

# ----------------------------
# Ensure required binary columns exist
# (If your dataset already has these, this won't overwrite them)
# ----------------------------
def to_yes_no_binary(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map({"no": 0, "yes": 1})

if "smoker_binary" not in df.columns:
    df["smoker_binary"] = to_yes_no_binary(df["smoker"])

if "diabetic_binary" not in df.columns:
    df["diabetic_binary"] = to_yes_no_binary(df["diabetic"])

if "gender_binary" not in df.columns:
    g = df["gender"].astype(str).str.strip().str.lower()
    df["gender_binary"] = g.map({"female": 0, "male": 1})

# ----------------------------
# Prepare ML dataset
# ----------------------------
feature_cols = ["age", "bmi", "bloodpressure", "smoker_binary", "diabetic_binary", "gender_binary"]
target_col = "claim"

df_ml = df.dropna(subset=feature_cols + [target_col]).copy()

# Safety check: ensure binary columns contain 0/1 (and not NaN)
for col in ["smoker_binary", "diabetic_binary", "gender_binary"]:
    df_ml = df_ml[df_ml[col].isin([0, 1])]

st.write(f"Rows used for ML: **{len(df_ml):,}**")

X = df_ml[feature_cols]
y = df_ml[target_col]

# ----------------------------
# Train/test split + train model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Predict + evaluate
# ----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))  # version-safe RMSE

col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("RMSE (£)", f"{rmse:,.2f}")

st.markdown("""
**How to interpret these:**
- **R²** shows how much of the variation in claims the model can explain (higher is better).
- **RMSE** is the typical prediction error in pounds (£) (lower is better).
""")

st.divider()

# ----------------------------
# Actual vs Predicted chart
# ----------------------------
st.subheader("Actual vs Predicted Claims (Test Set)")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.6)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")  # ideal line

ax.set_xlabel("Actual Claim (£)")
ax.set_ylabel("Predicted Claim (£)")
ax.set_title("Actual vs Predicted Insurance Claims")
st.pyplot(fig)

st.markdown("""
**Interpretation:**  
Points closer to the dashed diagonal line represent more accurate predictions.
""")

st.divider()

# ----------------------------
# Show coefficients (optional but great for learning)
# ----------------------------
st.subheader("Feature Influence (Model Coefficients)")

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

st.dataframe(coef_df, use_container_width=True)

st.caption("Positive coefficients increase predicted claim amounts; negative coefficients decrease them (holding other features constant).")

st.divider()

# ----------------------------
# Prediction form (simple demo)
# ----------------------------
st.subheader("Try a Prediction")

age = st.slider("Age", 18, 80, 40)
bmi = st.slider("BMI", 15.0, 45.0, 25.0)
bp = st.slider("Blood Pressure", 80, 180, 120)
smoker = st.selectbox("Smoker", ["no", "yes"])
diabetic = st.selectbox("Diabetic", ["no", "yes"])
gender = st.selectbox("Gender", ["female", "male"])

input_df = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "bloodpressure": [bp],
    "smoker_binary": [1 if smoker == "yes" else 0],
    "diabetic_binary": [1 if diabetic == "yes" else 0],
    "gender_binary": [1 if gender == "male" else 0],
})

pred = model.predict(input_df)[0]
st.success(f"Estimated Insurance Claim: **£{pred:,.2f}**")
