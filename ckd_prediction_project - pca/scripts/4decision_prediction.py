import pandas as pd
import joblib
import numpy as np

def predict_ckd(input_data):
    # Feature names and categorical columns
    original_features = [
        "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc",
        "htn","dm","cad","appet","pe","ane"
    ]
    categorical_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]

    # Format as DataFrame
    X_manual = pd.DataFrame([input_data])

    # Encode categorical features
    X_manual_encoded = pd.get_dummies(X_manual, columns=categorical_cols)

    # Load scaler, PCA, and columns used during training
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
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
    model = joblib.load("models/dtree_pca_model.pkl")

    # Predict class (0 = Not CKD, 1 = CKD)
    pred = model.predict(X_manual_pca)
    return int(pred[0])

# Example usage:
input_data = {
    "age": 50, "bp": 80, "sg": 1.020, "al": 1, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 121, "bu": 36, "sc": 1.2, "sod": 137, "pot": 4.5, "hemo": 15.4,
    "pcv": 44, "wc": 7800, "rc": 5.2, "htn": "no", "dm": "no", "cad": "no",
    "appet": "good", "pe": "no", "ane": "no"
 }
print(predict_ckd(input_data))
