# Customer Churn Prediction Dashboard

This project is a complete end-to-end data analysis and machine learning solution designed to predict customer churn using a public dataset from Kaggle. It demonstrates all key stages of a real data science workflow — from data preprocessing to deployment — and culminates in an interactive Streamlit dashboard.

---

## Objective

The goal is to:
- Analyze customer behavior and factors leading to churn
- Build a machine learning model that predicts churn accurately
- Provide business insights through visual analytics
- Deploy a user-friendly dashboard for interaction and decision-making

---

## Dataset Source

- Name: Churn for Bank Customers  
- Source: [Kaggle – Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)  
- The dataset includes 10,000 entries with details like credit score, geography, balance, churn status, etc.

---

## Workflow Summary

## 1. Data Cleaning & Preprocessing
- Removed duplicates and constant features
- Handled outliers using capping (IQR method)
- Addressed skewness with log transformation
- Created new feature flags like `ZeroBalanceFlag`

## 2. Feature Engineering
- Label encoding for binary categorical variables
- One-hot encoding for multi-class categorical features (e.g., Geography)
- Log transformation for skewed numerical features (e.g., Balance)

## 3. Class Imbalance Handling
- Detected class imbalance in the target variable `Exited`
- Applied SMOTE (Synthetic Minority Over-sampling Technique)

## 4. Model Building
- Trained Logistic Regression as a baseline model — noticed poor recall due to class imbalance
- Applied SMOTE to rebalance the training dataset
- Trained a Random Forest Classifier on the balanced data — achieved significant improvement in recall and F1-score

## 5. Interactive Streamlit Dashboard
- Dashboard built using Streamlit to:
    - Pie chart of churn vs retained customers
    - Bar chart of churn count by geography
    - Filters by gender and geography to customize the view
    - A live form to input new customer details and predict churn

---

## Tech Stack

- Language: Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit
- Modeling: Logistic Regression, SMOTE
- Deployment: Streamlit App (local / cloud)

---

## 📁 Project Structure

 File / Folder                  Description
 app.py                         Streamlit code for the interactive dashboard
 churn_prediction_model.pkl     Trained Random Forest model used for prediction
 final_predictions.csv          Final dataset with predicted churn values
 cleaned_after_outliers.csv     Dataset after outlier treatment
 requirements.txt               List of all required Python libraries
 README.md                      Detailed overview of the project
