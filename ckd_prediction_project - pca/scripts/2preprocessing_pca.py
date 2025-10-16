import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

# 1. Load the dataset
df = pd.read_csv("data/kidney_disease_mmm.csv")

# 2. Strip whitespace from strings
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 3. Drop rows with missing target
df = df[df["classification"].notna()]
df["classification"] = df["classification"].astype(int)

# 4. Separate features and target
X = df.drop("classification", axis=1)
y = df["classification"]

# 5. Encode categorical columns
cat_cols = X.select_dtypes(include="object").columns
X[cat_cols] = X[cat_cols].apply(LabelEncoder().fit_transform)

# 6. Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 7. Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# 8. Apply PCA (top 8 components)
pca = PCA(n_components=8)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=[f'pca_{i+1}' for i in range(8)])

# 9. Combine PCA features with target
df_pca = pd.concat([X_pca, y.reset_index(drop=True)], axis=1)

# 10. Save PCA-reduced dataset and objects
os.makedirs("data", exist_ok=True)
df_pca.to_csv("data/kidney_disease_pca.csv", index=False)
joblib.dump(pca, "models/pca_transform.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(imputer, "models/imputer.pkl")

print("PCA with 8 components saved to 'data/kidney_disease_pca.csv'")

