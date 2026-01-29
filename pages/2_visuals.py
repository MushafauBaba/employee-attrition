import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧪 Hypotheses & Visual Results")

df = pd.read_csv("data/insurance_data_clean.csv")

st.markdown("""
This page explores three hypotheses about factors that may influence **insurance claim amounts**.
Use the filters in the sidebar to explore specific groups of patients.
""")

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

# Region filter
regions = sorted(df["region"].dropna().unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

# Smoker filter
smoker_values = sorted(df["smoker"].dropna().unique())
selected_smoker = st.sidebar.multiselect("Smoker", smoker_values, default=smoker_values)

# Diabetic filter
diabetic_values = sorted(df["diabetic"].dropna().unique())
selected_diabetic = st.sidebar.multiselect("Diabetic", diabetic_values, default=diabetic_values)

# Apply filters
df_filtered = df[
    (df["region"].isin(selected_regions)) &
    (df["smoker"].isin(selected_smoker)) &
    (df["diabetic"].isin(selected_diabetic))
].copy()

st.sidebar.write(f"Rows after filters: {len(df_filtered):,}")

st.divider()

# ----------------------------
# Hypothesis 1
# ----------------------------
st.subheader("Hypothesis 1: Smokers have higher claims than non-smokers")

st.markdown("""
**Interpretation:**  
This box plot compares the distribution of insurance claim amounts between smokers and non-smokers.
A higher median and wider spread for smokers suggests increased claim amounts and higher variability.
""")

fig1, ax1 = plt.subplots(figsize=(9, 4))
sns.boxplot(data=df_filtered, x="claim", y="smoker", ax=ax1)
ax1.set_title("Insurance Claim Amounts by Smoking Status")
ax1.set_xlabel("Claim Amount")
ax1.set_ylabel("Smoker")
st.pyplot(fig1)

st.divider()

# ----------------------------
# Hypothesis 2
# ----------------------------
st.subheader("Hypothesis 2: High BMI + Diabetes increases claims")

st.markdown("""
**Interpretation:**  
This heatmap shows the **average claim amount** for each combination of BMI category and diabetic status.
Higher values among obese and diabetic patients suggest a compounding effect on claims.
""")

pivot = df_filtered.pivot_table(
    values="claim",
    index="diabetic",
    columns="bmi_category",
    aggfunc="mean"
)

fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.heatmap(pivot, annot=True, fmt=".0f", ax=ax2)
ax2.set_title("Average Insurance Claim by BMI Category and Diabetic Status")
ax2.set_xlabel("BMI Category")
ax2.set_ylabel("Diabetic")
st.pyplot(fig2)

st.divider()

# ----------------------------
# Hypothesis 3
# ----------------------------
st.subheader("Hypothesis 3: Gender has minimal impact on claims")

st.markdown("""
**Interpretation:**  
This bar chart compares the **average claim amount** for male and female patients.
Similar averages suggest there is no meaningful difference in claims by gender in this dataset.
""")

gender_mean = df_filtered.groupby("gender")["claim"].mean()

fig3, ax3 = plt.subplots(figsize=(6, 4))
gender_mean.plot(kind="bar", ax=ax3)
ax3.set_title("Average Insurance Claim by Gender")
ax3.set_xlabel("Gender")
ax3.set_ylabel("Average Claim")
st.pyplot(fig3)
