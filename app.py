# import streamlit as st
# import pandas as pd
# import joblib

# # Load model
# model = joblib.load('churn_prediction_model.pkl')

# st.set_page_config(page_title="Churn Dashboard", layout="wide")  # optional, improves layout
# st.title("📊 Customer Churn Prediction Dashboard")
# st.markdown("Use this dashboard to explore churn insights and predict customer behavior.")


# # Input form
# st.sidebar.header("Customer Information")
# CreditScore = st.sidebar.number_input("Credit Score", 300, 900, 600)
# Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
# Age = st.sidebar.slider("Age", 18, 100, 30)
# Tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
# NumOfProducts = st.sidebar.slider("Number of Products", 1, 4, 2)
# HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
# IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])
# EstimatedSalary = st.sidebar.number_input("Estimated Salary", 0, 200000, 50000)
# Log_Balance = st.sidebar.slider("Log Balance", 0.0, 15.0, 11.0)
# ZeroBalanceFlag = st.sidebar.selectbox("Zero Balance Flag", [0, 1])
# Geography = st.sidebar.selectbox("Geography", ['Germany', 'Spain'])

# # Encoding Gender & Geography
# Gender = 1 if Gender == 'Male' else 0
# Geography_Germany = 1 if Geography == 'Germany' else 0
# Geography_Spain = 1 if Geography == 'Spain' else 0

# # Final input
# input_data = pd.DataFrame([[CreditScore, Gender, Age, Tenure, NumOfProducts,
#                             HasCrCard, IsActiveMember, EstimatedSalary, 
#                             ZeroBalanceFlag, Log_Balance,
#                             Geography_Germany, Geography_Spain]],
#                           columns=['CreditScore', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 
#                                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
#                                    'ZeroBalanceFlag', 'Log_Balance',
#                                    'Geography_Germany', 'Geography_Spain'])

# # Prediction
# if st.button("Predict"):
#     prediction = model.predict(input_data)[0]
#     result = "Exited" if prediction == 1 else "Retained"
#     st.success(f"The customer is likely to **{result}**.")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Page Config ---
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# --- Load Model & Data ---
model = joblib.load("churn_prediction_model.pkl")
df = pd.read_csv("final_predictions.csv")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #003366;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)

# --- Sidebar Filters ---
with st.sidebar:
    st.header("🔍 Filter Data")
    selected_gender = st.multiselect("Gender", df['Gender'].unique(), default=df['Gender'].unique())
    selected_geo = st.multiselect("Geography", ['Germany', 'Spain'], default=['Germany', 'Spain'])

# --- Filtered Dataset ---
filtered_df = df[
    (df['Gender'].isin(selected_gender)) &
    (
        ((df['Geography_Germany'] == 1) & ('Germany' in selected_geo)) |
        ((df['Geography_Spain'] == 1) & ('Spain' in selected_geo))
    )
]

# --- KPI Section ---
st.markdown("### 📌 Key Metrics")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Customers", len(filtered_df))
kpi2.metric("Exited Customers", (filtered_df['Predicted_Churn'] == 'Exited').sum())
kpi3.metric("Retention Rate", f"{100 - ((filtered_df['Predicted_Churn'] == 'Exited').mean() * 100):.2f}%")

st.markdown("---")

# --- Visuals: Pie + Histogram ---
st.markdown("### 📈 Churn Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Churn Distribution")
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    filtered_df['Predicted_Churn'].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=90, colors=["skyblue", "salmon"], ax=ax1, textprops={'fontsize': 10}
    )
    ax1.set_ylabel('')
    st.pyplot(fig1)

with col2:
    st.markdown("#### Age Distribution of Exited Customers")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(
        filtered_df[filtered_df['Predicted_Churn'] == 'Exited']['Age'],
        bins=20, kde=True, color='salmon', ax=ax2
    )
    ax2.set_xlabel("Age")
    st.pyplot(fig2)

# --- Geography Bar Chart ---
st.markdown("### 🌍 Churn by Geography")
col_geo1, col_geo2, col_geo3 = st.columns([1, 2, 1])  # center the chart in middle column

with col_geo2:
    fig3, ax3 = plt.subplots(figsize=(4, 3))  # reduced actual size
    geo_data = filtered_df[filtered_df['Predicted_Churn'] == 'Exited'].copy()
    geo_data['Geography'] = geo_data[['Geography_Germany', 'Geography_Spain']].apply(
        lambda row: 'Germany' if row['Geography_Germany'] == 1 else 'Spain', axis=1
    )
    sns.countplot(x='Geography', data=geo_data, palette='Set2', ax=ax3)
    ax3.set_title("Exited Customers by Geography", fontsize=12)
    st.pyplot(fig3)


st.markdown("---")

# --- Prediction Section ---
st.markdown("### 🎯 Predict Churn for New Customer")

with st.form(key="prediction_form"):
    left, right = st.columns(2)

    with left:
        CreditScore = st.slider("Credit Score", 300, 900, 600)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.slider("Age", 18, 100, 40)
        Tenure = st.slider("Tenure (Years)", 0, 10, 3)
        NumOfProducts = st.slider("Number of Products", 1, 4, 2)

    with right:
        HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
        IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
        EstimatedSalary = st.number_input("Estimated Salary", min_value=1000.0, value=50000.0)
        balance = st.number_input("Enter Balance", min_value=0.0, format="%.2f")
        # Log_Balance = st.slider("Log Balance", 0.0, 15.0, 11.0)
        # ZeroBalanceFlag = st.selectbox("Zero Balance Flag", [0, 1])
        Geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])

    submit = st.form_submit_button("🔎 Predict")

# --- Prediction Logic ---
# --- Prediction Logic ---
if submit:
    Gender_val = 1 if Gender == "Male" else 0
    Geo_Germany = 1 if Geography == "Germany" else 0
    Geo_Spain = 1 if Geography == "Spain" else 0

    # Derived features
    ZeroBalanceFlag = 1 if balance == 0 else 0
    Log_Balance = np.log(balance + 1)

    input_data = pd.DataFrame([[  # keep all inputs aligned to model
        CreditScore, Gender_val, Age, Tenure, NumOfProducts, HasCrCard,
        IsActiveMember, EstimatedSalary, ZeroBalanceFlag, Log_Balance,
        Geo_Germany, Geo_Spain
    ]], columns=[
        'CreditScore', 'Gender', 'Age', 'Tenure', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'ZeroBalanceFlag', 'Log_Balance', 'Geography_Germany', 'Geography_Spain'
    ])

    prediction = model.predict(input_data)[0]
    result = "❌ Likely to EXIT" if prediction == 1 else "✅ Likely to RETAIN"
    st.success(f"Prediction: **{result}**")


# --- Footer ---
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:14px;'> Deepali Sharma | Customer Churn Analysis Dashboard</p>", unsafe_allow_html=True)
