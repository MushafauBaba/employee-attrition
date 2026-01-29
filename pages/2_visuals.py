import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧪 Visualisations")

df = pd.read_csv("data/processed_attrition_dataset.csv")


# ----------------------------
# Attrition Percentage of Leavers
# ----------------------------
st.subheader("Percentage of Employee Attrition")

st.markdown("""
**Interpretation:**  
The below charts shows us that 16% of the 1470 employees recorded during the analysis decided to leave for a particular reason.  
""")


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

Risk_counts = df['Attrition'].value_counts()

colors = ['#2ecc71', '#FF0000']
axes[0].pie(Risk_counts, labels=['No','Yes'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('Attrition Risk Distribution')

axes[1].bar(['No','Yes'], Risk_counts, color=colors)
axes[1].set_ylabel('Count')
axes[1].set_title('Attrition Count')

st.pyplot(fig)
st.divider()

# ----------------------------
# Feature Correlation Matrix
# ----------------------------
st.subheader("Correlation Analysis of Features")

st.markdown("""
**Interpretation:**  
During the EDA process we wanted to see if there was any correlation between the following key features:
- Performance Rating and Number of Trainings
- Salary Hike Percent and Job Satisfaction

Based on the below heatmap the analysis shows that there is no correlation or relationship existing between Performance rating and the number of training in the last year.
There is also no correlation between Salary hike and Job satisfaction.
However, there is a strong 0.77 correlation coefficient between Salary Hike percent and Perfomance Rating,
which means there is strong linear relationship between salary hike and performance rating. Generally, higher salary hike results to higher performance rating and vice versa
""")

num_cols = df.select_dtypes(include="number").columns
correlation_matrix = df[num_cols].corr()

fig2, ax2 = plt.subplots(figsize=(12, 10))

sns.heatmap(correlation_matrix, annot=False, cmap='RdBu_r', center=0, 
            square=True, linewidths=1, ax=ax2)
ax2.set_title('Feature Correlation Matrix', fontsize=14)
st.pyplot(fig2, use_container_width=True)

st.divider()