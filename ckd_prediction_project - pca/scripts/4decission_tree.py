import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load original dataset
df_orig = pd.read_csv("data/kidney_disease_mmm.csv")
original_features = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"
]

# Separate features and target
X_orig = df_orig[original_features]
y = df_orig["classification"]

# Encode categorical features
categorical_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
X_encoded = pd.get_dummies(X_orig, columns=categorical_cols)

# Fit scaler and PCA on original features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

# Save scaler and PCA
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")

# Use PCA-transformed features for model training
X = pd.DataFrame(X_pca, columns=[f"pca_{i+1}" for i in range(8)])

# Check class balance
print("Class distribution:")
print(y.value_counts())


# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []
os.makedirs("confusion_matrices", exist_ok=True)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(f"Classification Report (Fold {fold}):\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Decision Tree (Fold {fold})')
    plt.tight_layout()
    plt.savefig(f"confusion_matrices/dtree_fold{fold}.png")
    plt.close()
    fold += 1

# Save final model and scaler (trained on all data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_scaled, y)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/dtree_pca_model.pkl")
joblib.dump(scaler, "models/dtree_pca_scaler.pkl")


# Visualization of fold accuracies
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(accuracies)+1), accuracies, color='green', alpha=0.7)
plt.axhline(np.mean(accuracies), color='red', linestyle='--', label=f'Average Accuracy ({np.mean(accuracies):.4f})')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Decision Tree K-Fold Accuracies')
plt.xticks(range(1, len(accuracies)+1))
plt.legend()
plt.tight_layout()
plt.show()

print("\nAverage Accuracy:", np.mean(accuracies))
