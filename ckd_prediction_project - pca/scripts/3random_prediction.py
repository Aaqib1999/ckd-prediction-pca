import pandas as pd
import joblib
import numpy as np

def predict_ckd_rf(input_data):
    original_features = [
        "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc",
        "htn","dm","cad","appet","pe","ane"
    ]
    categorical_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]

    X_manual = pd.DataFrame([input_data])
    X_manual_encoded = pd.get_dummies(X_manual, columns=categorical_cols)

    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    X_train_cols = scaler.feature_names_in_

    for col in X_train_cols:
        if col not in X_manual_encoded.columns:
            X_manual_encoded[col] = 0
    X_manual_encoded = X_manual_encoded[X_train_cols]
    X_manual_encoded = X_manual_encoded.astype(float)

    X_manual_scaled = scaler.transform(X_manual_encoded)
    X_manual_pca = pca.transform(X_manual_scaled)

    model = joblib.load("models/random_forest.pkl")
    pred = model.predict(X_manual_pca)
    return int(pred[0])

# Example usage:
# input_data = {
#     "age": 50, "bp": 80, "sg": 1.020, "al": 1, "su": 0,
#     "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
#     "bgr": 121, "bu": 36, "sc": 1.2, "sod": 137, "pot": 4.5, "hemo": 15.4,
#     "pcv": 44, "wc": 7800, "rc": 5.2, "htn": "no", "dm": "no", "cad": "no",
#     "appet": "good", "pe": "no", "ane": "no"
#}
#print(predict_ckd_rf(input_data))
input_data = {
    "age": 65, "bp": 150, "sg": 1.005, "al": 4, "su": 3,
    "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "present",
    "bgr": 300, "bu": 100, "sc": 5.0, "sod": 120, "pot": 6.0, "hemo": 8.0,
    "pcv": 25, "wc": 18000, "rc": 2.5, "htn": "yes", "dm": "yes", "cad": "yes",
    "appet": "poor", "pe": "yes", "ane": "yes"
}
print(predict_ckd_rf(input_data))