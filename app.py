import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ===============================
# Helper Functions
# ===============================
def prepare_data(data, target_column):
    """Prepare data by scaling and splitting."""
    data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    data = data.fillna(data.mean())  # Handle missing values
    X = data.drop(columns=[target_column])  # Features (exclude target column)
    y = data[target_column]  # Target (default column)
    
    # Standardize the data (feature scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_retail_model(data):
    """Train logistic regression model for retail portfolio."""
    X_train, X_test, y_train, y_test = prepare_data(data, 'default')
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def train_corporate_model(data):
    """Train random forest model for corporate portfolio."""
    X_train, X_test, y_train, y_test = prepare_data(data, 'default')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# ===============================
# Streamlit App
# ===============================

st.title("Credit Rating Model Application")
st.sidebar.header("Portfolio Options")

# Select Portfolio Type
portfolio_type = st.sidebar.selectbox("Select Portfolio Type", ["Retail", "Corporate"])

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    if 'default' in data.columns:
        if portfolio_type == "Retail":
            st.subheader("Training Logistic Regression Model for Retail Portfolio")
            model, accuracy = train_retail_model(data)
            st.write("Model Accuracy:", accuracy)
        elif portfolio_type == "Corporate":
            st.subheader("Training Random Forest Model for Corporate Portfolio")
            model, accuracy = train_corporate_model(data)
            st.write("Model Accuracy:", accuracy)
    else:
        st.error("The dataset must contain a 'default' column for training.")

# User Input for Prediction
if st.sidebar.checkbox("Enter New Data for Prediction"):
    st.subheader("Enter Details for Prediction")

    if portfolio_type == "Retail":
        income = st.number_input("Customer Income", min_value=0.0, value=50000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)

        if st.button("Predict"):
            features = np.array([[income, loan_amount, age]])
            prediction = model.predict(features)
            st.write("Prediction:", "Default" if prediction[0] == 1 else "No Default")

    elif portfolio_type == "Corporate":
        assets = st.number_input("Total Assets", min_value=0.0, value=100000.0)
        liabilities = st.number_input("Total Liabilities", min_value=0.0, value=50000.0)
        revenue = st.number_input("Total Revenue", min_value=0.0, value=150000.0)

        if st.button("Predict"):
            features = np.array([[assets, liabilities, revenue]])
            prediction = model.predict(features)
            st.write("Prediction:", "Default" if prediction[0] == 1 else "No Default")
