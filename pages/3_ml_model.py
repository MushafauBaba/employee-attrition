import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

st.title("🤖 Machine Learning: Clustering Employees into Risk Attrition Categories")

st.markdown("""
The goal of this page is measure the risk level (from low risk to high risk), of employee attrition using a machine learning model.
""")
