import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# Load original dataset
df_orig = pd.read_csv("data/kidney_disease_mmm.csv")
original_features = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"
]
categorical_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
X_orig = df_orig[original_features]
y = df_orig["classification"]
X_encoded = pd.get_dummies(X_orig, columns=categorical_cols)

# Load scaler and PCA (fit if not present)
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
if os.path.exists("models/scaler.pkl") and os.path.exists("models/pca.pkl"):
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    X_scaled = scaler.transform(X_encoded)
    X_pca = pca.transform(X_scaled)
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_scaled)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")

# Use PCA-transformed features for model training
X = pd.DataFrame(X_pca, columns=[f"pca_{i+1}" for i in range(8)])


# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []
os.makedirs("confusion_matrices", exist_ok=True)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = (y_pred == y_test).mean()
    accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(f"Classification Report (Fold {fold}):\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Random Forest (Fold {fold})')
    plt.tight_layout()
    plt.savefig(f"confusion_matrices/random_forest_fold{fold}.png")
    plt.close()
    fold += 1

# Save final model (trained on all data)
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest.pkl")

# Visualization of fold accuracies
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(accuracies)+1), accuracies, color='darkgreen', alpha=0.7)
plt.axhline(np.mean(accuracies), color='red', linestyle='--', label=f'Average Accuracy ({np.mean(accuracies):.4f})')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Random Forest K-Fold Accuracies')
plt.xticks(range(1, len(accuracies)+1))
plt.legend()
plt.tight_layout()
plt.show()

print("\nAverage Accuracy:", np.mean(accuracies))
