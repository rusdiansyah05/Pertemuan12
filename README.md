build_embeddings.py
Kode Program
# build_embeddings.py
import os
import glob
import numpy as np
from tqdm import tqdm
from utils_facenet import embed_from_path

def iter_images(root):
    """Generator untuk iterasi semua gambar dalam folder"""
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    for cls in classes:
        for p in glob.glob(os.path.join(root, cls, "*")):
            yield p, cls

def build_matrix(root):
    X, y, bad = [], [], []
    
    for path, cls in tqdm(list(iter_images(root))):
        emb = embed_from_path(path)
        if emb is None:
            bad.append(path)
            continue
        X.append(emb)
        y.append(cls)
    
    return np.array(X), np.array(y), bad

if __name__ == "__main__":
    X, y, bad = build_matrix("data/train")
    print ("Embeddings:", X.shape, "Labels:", y.shape, "gagal Deteksi:", len(bad))
    np.save("X_train.npy", X ); np.save("y_train.npy",y)

Analisis
Kode build_embeddings.py berfungsi untuk mengekstraksi embedding wajah dari dataset pelatihan secara otomatis dengan membaca setiap folder kelas di dalam direktori data/train, kemudian memproses seluruh gambar menggunakan fungsi embed_from_path untuk menghasilkan embedding berukuran 512 dimensi; gambar yang tidak terdeteksi wajahnya akan dicatat dalam daftar bad, sedangkan embedding yang valid disimpan dalam array X dan labelnya disimpan dalam array y, sebelum akhirnya kedua array tersebut disimpan sebagai file X_train.npy dan y_train.npy, serta program menampilkan jumlah total embedding, label, dan gambar yang gagal diproses untuk memastikan dataset siap digunakan pada tahap pelatihan model pengenalan wajah seperti KNN atau SVM.


eval_folder.py
Kode Program
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

Analisis
Kode eval_folder.py digunakan untuk mengevaluasi performa model FaceNet + SVM dengan cara membaca seluruh gambar dalam folder validasi data/val, menghitung embedding menggunakan embed_from_path, lalu memprediksi kelas menggunakan model SVM yang sebelumnya dimuat dari facenet_svm.joblib. Setiap prediksi dibandingkan dengan label folder aslinya untuk menghitung akurasi keseluruhan, sekaligus mencatat performa per kelas melalui struktur defaultdict. Untuk setiap gambar yang berhasil diekstraksi embedding-nya, program menyimpan label benar (Y_true) dan label prediksi (Y_pred), lalu menghitung akurasi global menggunakan perbandingan array boolean, serta menampilkan akurasi per kelas dalam format jumlah benar, total sampel, dan rasio keberhasilan. Dengan demikian, skrip ini memberikan gambaran kuantitatif seberapa akurat model mengenali wajah pada dataset uji berdasarkan struktur folder validasi yang kamu gunakan.

predict_one.py
Kode Program
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

Analisis
Kode ini berfungsi untuk melakukan prediksi wajah pada satu gambar menggunakan model SVM yang sebelumnya dilatih dan disimpan dalam file facenet_svm.joblib. Proses dimulai dengan memuat embedding dari gambar melalui fungsi embed_from_path, dan apabila tidak ada wajah yang terdeteksi maka fungsi langsung mengembalikan label "NO_FACE" dengan confidence 0.0. Jika embedding berhasil diperoleh, embedding diratakan dan dibentuk ulang menjadi vektor 2D sebelum dimasukkan ke model. Prediksi kelas dilakukan dengan clf.predict, sedangkan nilai confidence dihitung menggunakan clf.predict_proba, di mana nilai probabilitas tertinggi dianggap sebagai tingkat keyakinan model. Pada bagian utama, script menguji satu gambar contoh dan menampilkan label hasil prediksi beserta confidence, sehingga kode ini sangat berguna untuk melakukan pengecekan cepat terhadap kualitas model FaceNet + SVM pada satu gambar secara individual.

train_classifier.py
Kode Program
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

Analisis
Kode train_classifier.py digunakan untuk melatih model klasifikasi wajah menggunakan SVM dengan kernel RBF pada dataset yang sudah diekstraksi embedding-nya dan disimpan dalam file X_train.npy dan y_train.npy. Model SVM dibangun dengan menggunakan pipeline, yang pertama-tama melakukan standarisasi fitur dengan StandardScaler sebelum diteruskan ke SVM dengan parameter yang telah disesuaikan (C=10, gamma="scale", dan class_weight="balanced"). Setelah pelatihan selesai, model disimpan dalam file facenet_svm.joblib menggunakan joblib.dump. Akhirnya, akurasi pelatihan dihitung secara manual dengan membandingkan prediksi model terhadap label yang sebenarnya dan mencetak nilai akurasi, memberikan gambaran langsung tentang seberapa baik model bekerja pada data pelatihan yang ada.

train_knn.py
Kode Program
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

Analisis
Kode train_knn.py digunakan untuk melatih model K-Nearest Neighbors (KNN) pada dataset embedding wajah yang telah disimpan dalam file X_train.npy dan y_train.npy. Model KNN dibangun menggunakan pipeline yang pertama-tama melakukan standarisasi fitur dengan StandardScaler sebelum diteruskan ke klasifikasi KNN dengan parameter jumlah tetangga (k=3) dan menggunakan metrik jarak Euclidean. Setelah proses pelatihan selesai, model yang telah dilatih disimpan dalam file facenet_knn.joblib menggunakan joblib.dump, memungkinkan model untuk digunakan pada tahap prediksi tanpa perlu melatih ulang.

utils_facenet.py
Kode Program
# utils_facenet.py
import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan device: {device}")

# Detector & aligner
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)

# Embedder (512-dim)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def read_img_bgr(path):
    img = cv2.imread(path)  # BGR
    if img is None:
        raise ValueError(f"Gagal baca: {path}")
    return img

def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

@torch.no_grad()
def face_align(img_bgr):
    """Return aligned face as PIL.Image (160x160) or None if not found."""
    pil = bgr_to_pil(img_bgr)
    aligned = mtcnn(pil)  # tensor [3,160,160] or None
    return aligned

@torch.no_grad()
def embed_face_tensor(face_tensor):
    """face_tensor: torch.Tensor [3,160,160] in range [0,1] (from MTCNN)"""
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(device)  # [1,3,160,160]
    emb = embedder(face_tensor)  # [1,512]
    return emb.squeeze(0).cpu().numpy()  # (512,)

@torch.no_grad()
def embed_from_path(path):
    img = read_img_bgr(path)
    face = face_align(img)
    if face is None:
        return None
    return embed_face_tensor(face)

def cosine_similarity(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))

Analisis
Kode utils_facenet.py berfungsi untuk mempersiapkan dan mengekstraksi fitur wajah menggunakan model FaceNet. Dimulai dengan memuat model MTCNN untuk mendeteksi dan menyelaraskan wajah, serta model InceptionResnetV1 sebagai embedder untuk menghasilkan embedding wajah berukuran 512 dimensi. Proses dimulai dengan membaca gambar menggunakan OpenCV (read_img_bgr), lalu mengonversi gambar dari format BGR ke RGB. Setelah itu, wajah pada gambar diselaraskan menggunakan MTCNN (face_align). Jika wajah terdeteksi, gambar yang diselaraskan dikirimkan ke embedder untuk menghasilkan embedding menggunakan metode embed_face_tensor. Fungsi embed_from_path menggabungkan proses tersebut untuk mengekstraksi embedding wajah dari gambar berdasarkan path file. Selain itu, fungsi cosine_similarity digunakan untuk menghitung kesamaan antara dua embedding wajah dengan menghitung nilai dot product setelah normalisasi, yang dapat digunakan untuk membandingkan wajah dalam sistem pengenalan wajah.

verify_cli.py
Kode Program
# verify_cli.py
import argparse
from utils_facenet import embed_from_path, cosine_similarity

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verifikasi wajah 1:1")
    ap.add_argument("img1", help="Path gambar pertama")
    ap.add_argument("img2", help="Path gambar kedua")
    ap.add_argument("--th", type=float, default=0.85, help="Threshold similarity")
    args = ap.parse_args()

    e1 = embed_from_path(args.img1)
    e2 = embed_from_path(args.img2)
    
    if e1 is None or e2 is None:
        print("Wajah tidak terdeteksi pada salah satu gambar.")
    else:
        sim = cosine_similarity(e1, e2)
        match = "MATCH" if sim >= args.th else "NO MATCH"
        print(f"Similarity: {sim:.4f} -> {match} (th={args.th})")

Analisis
Kode verify_cli.py digunakan untuk melakukan verifikasi wajah 1:1 dengan membandingkan dua gambar yang diberikan melalui argumen command-line. Menggunakan argparse, pengguna dapat memberikan dua gambar melalui path (img1 dan img2), serta optional threshold (--th) untuk menentukan batas kesamaan wajah. Kode ini kemudian memuat embedding wajah dari kedua gambar menggunakan fungsi embed_from_path, dan jika salah satu gambar tidak memiliki wajah yang terdeteksi, maka akan muncul pesan error. Jika kedua gambar memiliki wajah yang terdeteksi, kode akan menghitung kesamaan antar embedding dengan menggunakan fungsi cosine_similarity. Berdasarkan nilai kesamaan yang dihitung, sistem akan memberikan hasil "MATCH" atau "NO MATCH" sesuai dengan threshold yang ditentukan, yang memungkinkan untuk mengevaluasi apakah kedua wajah tersebut berasal dari orang yang sama atau tidak.

verify_pair.py
Kode Program
# verify_pair.py
from utils_facenet import embed_from_path, cosine_similarity

# SESUAIKAN DENGAN STRUKTUR ANDA
img1 = "data/train/rusdi/Rusdi_1.jpg"  # GANTI
img2 = "data/train/rusdi/Rusdi_2.jpg"  # GANTI

emb1 = embed_from_path(img1)
emb2 = embed_from_path(img2)

if emb1 is None or emb2 is None:
    print("Wajah tidak terdeteksi pada salah satu gambar.")
else:
    sim = cosine_similarity(emb1, emb2)
    print("Cosine similarity:", sim)
    # Threshold umum (awal): 0.8-0.9 (semakin tinggi = semakin ketat)
    threshold = 0.85
    print("Match?", "YA" if sim >= threshold else "TIDAK")

Analisis
Kode verify_pair.py berfungsi untuk memverifikasi apakah dua gambar wajah milik orang yang sama dengan membandingkan kesamaan fitur wajah menggunakan cosine similarity. Gambar pertama dan kedua dimuat menggunakan fungsi embed_from_path, yang mengekstraksi embedding wajah dari kedua gambar tersebut. Jika wajah tidak terdeteksi pada salah satu gambar, maka kode akan menampilkan pesan kesalahan. Namun, jika kedua gambar berhasil diproses, kode menghitung nilai cosine similarity antara dua embedding wajah yang dihasilkan dan membandingkannya dengan threshold yang telah ditentukan (dalam hal ini, 0.85). Berdasarkan hasil perbandingan ini, kode akan mencetak hasil verifikasi berupa "YA" jika gambar memiliki kesamaan tinggi (match) atau "TIDAK" jika kesamaan rendah (no match).


