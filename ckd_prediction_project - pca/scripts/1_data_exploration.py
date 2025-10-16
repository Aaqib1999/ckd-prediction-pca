import pandas as pd
import joblib

# Load saved Logistic Regression model
model = joblib.load("models/logreg_pca_model.pkl")

# Example input (all PCA features required, update values as needed)
new_data = {
    'pca_1': [1.23],
    'pca_2': [0.45],
    'pca_3': [-0.67],
    'pca_4': [0.12],
    'pca_5': [-0.34],
    'pca_6': [0.01],
    'pca_7': [0.02],
    'pca_8': [-0.11]
}

input_df = pd.DataFrame(new_data)
prediction = model.predict(input_df)

print("Prediction:", "CKD" if prediction[0] == 1 else "Not CKD")
