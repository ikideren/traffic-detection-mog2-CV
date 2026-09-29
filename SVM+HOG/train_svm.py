import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import time

# 1. Memuat Data Fitur dan Label
print("Memuat data dari hog_features_labels.pkl...")
X, y = joblib.load('hog_features_labels.pkl')
print(f"Total sampel: {len(X)}")

# 2. Pembagian Data (Train & Test Split)
# Membagi 80% data untuk training dan 20% untuk pengujian (validasi)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data latih: {len(X_train)} sampel | Data uji: {len(X_test)} sampel")

# 3. Inisialisasi dan Pelatihan Model SVM
# Menggunakan LinearSVC karena lebih cepat dan cocok untuk fitur berdimensi tinggi (HOG)
svm_model = LinearSVC(C=0.05, max_iter=10000, dual=False, class_weight='balanced', random_state=42)

print("Mulai melatih model SVM... (Ini mungkin memakan waktu beberapa saat)")
start_time = time.time()
svm_model.fit(X_train, y_train)
end_time = time.time()
print(f"Pelatihan selesai dalam {end_time - start_time:.2f} detik.")

# 4. Evaluasi Model
print("Mengevaluasi model pada data uji...")
y_pred = svm_model.predict(X_test)

print("\n--- Hasil Evaluasi ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nLaporan Klasifikasi:")
# Target names disesuaikan dengan label (0: Background, 1: Kendaraan)
print(classification_report(y_test, y_pred, target_names=['Background', 'Kendaraan']))

# 5. Menyimpan Model yang Sudah Dilatih
joblib.dump(svm_model, 'svm_model.pkl')
print("\nModel berhasil disimpan sebagai svm_model.pkl!")