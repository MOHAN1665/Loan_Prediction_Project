import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load models
with open('random_forest_model.pkl', 'rb') as file:
    rf_clf = joblib.load(file)

with open('svm_model.pkl', 'rb') as file:
    svm_clf = joblib.load(file)

with open('knn_model.pkl', 'rb') as file:
    knn_clf = joblib.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    lr_clf = joblib.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = joblib.load(file)

# Streamlit app
st.title('Loan Prediction')

# Input features
no_of_dependents = st.selectbox('Number of Dependents', [0, 1, 2, 3, 4, 5])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
income_annum = st.number_input('Annual Income', value=0)
loan_amount = st.number_input('Loan Amount', value=0)
loan_term = st.number_input('Loan Term (in months)', value=360)
cibil_score = st.number_input('CIBIL Score', value=0)
residential_assets_value = st.number_input('Residential Assets Value', value=0)
commercial_assets_value = st.number_input('Commercial Assets Value', value=0)
luxury_assets_value = st.number_input('Luxury Assets Value', value=0)
bank_asset_value = st.number_input('Bank Asset Value', value=0)

# Create input data frame with the correct feature set
input_data = pd.DataFrame({
    'no_of_dependents': [no_of_dependents],
    'education': [education],
    'self_employed': [self_employed],
    'income_annum': [income_annum],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'luxury_assets_value': [luxury_assets_value],
    'bank_asset_value': [bank_asset_value]
})

# Encode categorical variables to match the training data
label_encoder = LabelEncoder()
input_data['education'] = label_encoder.fit_transform(input_data['education'])
input_data['self_employed'] = label_encoder.fit_transform(input_data['self_employed'])

# Debugging output to check encoded input values
st.write('Encoded Input Values:')
st.write(input_data)

# Select only the columns used in training
input_data = input_data[['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term',
                         'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                         'bank_asset_value']]

# Standardize numerical features using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Debugging output to check scaled input values
st.write('Input Data Scaled:')
st.write(input_data_scaled)

# Predict button
if st.button('Predict'):
    try:
        rf_pred = rf_clf.predict(input_data_scaled)[0]
        svm_pred = svm_clf.predict(input_data_scaled)[0]
        knn_pred = knn_clf.predict(input_data_scaled)[0]
        lr_pred = lr_clf.predict(input_data_scaled)[0]

        # Print predictions with custom messages
        st.write(f'Random Forest Prediction: {"loan is approved" if rf_pred == 0 else "loan not approved"}')
        st.write(f'SVM Prediction: {"loan is approved" if svm_pred == 0 else "loan not approved"}')
        st.write(f'K-Nearest Neighbors Prediction: {"loan is approved" if knn_pred == 0 else "loan not approved"}')
        st.write(f'Logistic Regression Prediction: {"loan is approved" if lr_pred == 0 else "loan not approved"}')


        # Print probabilities or decision function outputs if available
        if hasattr(rf_clf, 'predict_proba'):
            st.write(f'Random Forest Probabilities: {rf_clf.predict_proba(input_data_scaled)}')
        if hasattr(svm_clf, 'predict_proba'):
            st.write(f'SVM Probabilities: {svm_clf.predict_proba(input_data_scaled)}')
        if hasattr(knn_clf, 'predict_proba'):
            st.write(f'K-Nearest Neighbors Probabilities: {knn_clf.predict_proba(input_data_scaled)}')
        if hasattr(lr_clf, 'predict_proba'):
            st.write(f'Logistic Regression Probabilities: {lr_clf.predict_proba(input_data_scaled)}')

        # Determine loan approval based on predictions
        if rf_pred == 0 and svm_pred == 0 and knn_pred == 0 and lr_pred == 0:
            st.write('Loan Approved')
        else:
            st.write('Loan Not Approved')

    except ValueError as e:
        st.write(f"An error occurred: {e}")
