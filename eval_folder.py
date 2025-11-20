# eval_folder.py
import os, glob, numpy as np, joblib
from collections import defaultdict
from utils_facenet import embed_from_path

clf = joblib.load("facenet_svm.joblib")

def predict_emb(emb):
    proba = clf.predict_proba([emb])[0]
    idx = int(np.argmax(proba))
    return clf.classes_[idx], float(proba[idx])

# SESUAIKAN DENGAN STRUKTUR ANDA
root = "data/val"  # Pastikan ada gambar di folder val/rusdi/ dan val/aditya/

Y_true, Y_pred = [], []
per_cls = defaultdict(lambda: {"ok":0, "total":0})

for cls in sorted(os.listdir(root)):
    pdir = os.path.join(root, cls)
    if not os.path.isdir(pdir): continue
    for p in glob.glob(os.path.join(pdir, "*")):
        emb = embed_from_path(p)
        if emb is None: continue
        pred, conf = predict_emb(emb)
        Y_true.append(cls); Y_pred.append(pred)
        per_cls[cls]["total"] += 1
        per_cls[cls]["ok"] += int(pred == cls)

acc = np.mean([t==p for t,p in zip(Y_true, Y_pred)])
print("Accuracy:", acc)

for c, st in per_cls.items():
    if st["total"]>0:
        print(f"{c}: {st['ok']}/{st['total']} = {st['ok']/st['total']:.3f}")