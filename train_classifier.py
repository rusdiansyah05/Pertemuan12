# train_classifier.py - PERSIS MODUL TANPA CV
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

X = np.load("X_train.npy")
y = np.load("y_train.npy", allow_pickle=True)

print(f"Data training: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y)}")

# Pipeline: standardize -> SVM (RBF) - PERSIS MODUL
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"))
])

# LANGSUNG TRAINING TANPA CV - karena data sedikit
print("Training model...")
clf.fit(X, y)

# SIMPAN MODEL - PERSIS MODUL
joblib.dump(clf, "facenet_svm.joblib")
print("Model disimpan ke facenet_svm.joblib")

# Hitung accuracy training manual
train_pred = clf.predict(X)
accuracy = np.mean(train_pred == y)
print(f"Training accuracy: {accuracy:.4f}")