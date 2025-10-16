import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Feature names and categorical columns
original_features = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc",
    "htn","dm","cad","appet","pe","ane"
]
categorical_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]

# Categorical options
cat_options = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["notpresent", "present"],
    "ba": ["notpresent", "present"],
    "htn": ["no", "yes"],
    "dm": ["no", "yes"],
    "cad": ["no", "yes"],
    "appet": ["good", "poor"],
    "pe": ["no", "yes"],
    "ane": ["no", "yes"]
}

st.title("CKD Prediction App")

st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Decision Tree", "Logistic Regression", "Random Forest")
)

st.header("Enter Patient Data")
feature_ranges = {
    "age": (1, 100),
    "bp": (50, 200),
    "sg": (1.005, 1.025),
    "al": (0, 5),
    "su": (0, 5),
    "bgr": (22, 490),
    "bu": (1, 400),
    "sc": (0.4, 15.0),
    "sod": (100, 150),
    "pot": (2.5, 7.0),
    "hemo": (3.1, 17.8),
    "pcv": (10, 54),
    "wc": (2500, 25000),
    "rc": (2.1, 8.0)
}
input_data = {}
for feature in original_features:
    if feature in categorical_cols:
        input_data[feature] = st.selectbox(f"{feature}", cat_options[feature])
    else:
        min_val, max_val = feature_ranges.get(feature, (0.0, 100.0))
        val = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=min_val)
        # Enforce valid range
        if val < min_val:
            st.warning(f"{feature} value too low, setting to {min_val}")
            val = min_val
        elif val > max_val:
            st.warning(f"{feature} value too high, setting to {max_val}")
            val = max_val
        input_data[feature] = val

if st.button("Predict CKD"):
    # Format as DataFrame
    X_manual = pd.DataFrame([input_data])
    X_manual_encoded = pd.get_dummies(X_manual, columns=categorical_cols)

    # Load scaler, PCA, and columns used during training
    scaler = joblib.load("../models/scaler.pkl")
    pca = joblib.load("../models/pca.pkl")
    X_train_cols = scaler.feature_names_in_

    # Add missing columns (from training) as zeros
    for col in X_train_cols:
        if col not in X_manual_encoded.columns:
            X_manual_encoded[col] = 0
    X_manual_encoded = X_manual_encoded[X_train_cols]
    X_manual_encoded = X_manual_encoded.astype(float)

    # Scale and apply PCA
    X_manual_scaled = scaler.transform(X_manual_encoded)
    X_manual_pca = pca.transform(X_manual_scaled)

    # Load model
    if model_choice == "Decision Tree":
        model = joblib.load("../models/dtree_pca_model.pkl")
    elif model_choice == "Logistic Regression":
        model = joblib.load("../models/logistic_regression_model.pkl")
    else:
        model = joblib.load("../models/random_forest.pkl")

    pred = model.predict(X_manual_pca)
    st.success(f"Prediction: {'CKD (1)' if int(pred[0]) == 1 else 'Not CKD (0)'}")
