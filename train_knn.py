# train_knn.py
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
X = np.load("X_train.npy")
y = np.load("y_train.npy", allow_pickle=True)

print(f"Training KNN dengan {X.shape[0]} samples")

# Pipeline: standardize -> KNN
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3, metric="euclidean"))
])

clf.fit(X, y)
joblib.dump(clf, "facenet_knn.joblib")
print("Model KNN disimpan ke facenet_knn.joblib")