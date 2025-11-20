import joblib
from utils_facenet import embed_from_path
import numpy as np

# Load model
clf = joblib.load("facenet_svm.joblib")

def predict_image(path):
    emb = embed_from_path(path)
    if emb is None:
        return "NO_FACE", 0.0
    
    emb_flat = emb.flatten().reshape(1, -1)
    
    # FIX: Langsung pakai predict() saja
    prediction = clf.predict(emb_flat)[0]
    
    # Untuk confidence, pakai predict_proba
    proba = clf.predict_proba(emb_flat)[0]
    conf = max(proba)  # Confidence tertinggi
    
    return prediction, conf

if __name__ == "__main__":
    test_img = "data/train/rusdi/Rusdi_1.jpg"
    label, conf = predict_image(test_img)
    print(f"Prediksi: {label} (conf={conf:.3f})")