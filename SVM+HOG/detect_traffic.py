import cv2
import joblib
import numpy as np
from skimage.feature import hog
from imutils.object_detection import non_max_suppression

# 1. Konfigurasi Awal
MODEL_PATH = 'svm_model.pkl'
TEST_IMAGE = 'test_intersection.jpg'
IMG_SIZE = (64, 64)

# Menyetel kembali threshold ke angka seimbang setelah area pengganggu dibatasi
CONFIDENCE_THRESHOLD = 0.95 

print("Memuat model SVM...")
svm_model = joblib.load(MODEL_PATH)

print("Membaca citra uji...")
img = cv2.imread(TEST_IMAGE)

if img is None:
    raise FileNotFoundError(f"Error: Gambar '{TEST_IMAGE}' tidak ditemukan.")

img_copy = img.copy()
h_img, w_img, _ = img.shape

# --- ANCHOR SOLUSI OPTIMAL: STRICT LANE MASKING ---
# Kita buat poligon (masking) yang mengisolasi jalan raya saja.
pts = np.array([
    [int(w_img * 0.26), int(h_img * 0.22)],  # Kiri atas
    [int(w_img * 0.80), int(h_img * 0.22)],  # Kanan atas
    [int(w_img * 0.82), int(h_img * 0.95)],  # Kanan bawah
    [int(w_img * 0.20), int(h_img * 0.95)]   # Kiri bawah
], np.int32)

# PERBAIKAN: Menggunakan tuple (h_img, w_img) tanpa tanda petik
mask = np.zeros((h_img, w_img), dtype=np.uint8)
cv2.fillPoly(mask, [pts], 255)
img_masked = cv2.bitwise_and(img, img, mask=mask)

# Crop boundaries minimal untuk mempercepat Selective Search
ymin, ymax = int(h_img * 0.22), int(h_img * 0.95)
xmin, xmax = int(w_img * 0.20), int(w_img * 0.82)
img_roi = img_masked[ymin:ymax, xmin:xmax]

# Gunakan blur ringan untuk menghaluskan tekstur aspal sisa
img_roi_blur = cv2.GaussianBlur(img_roi, (3, 3), 0)

# 2. Inisialisasi Selective Search pada area jalan terisolasi
print("Menjalankan Selective Search pada koridor lalu lintas...")
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img_roi_blur)
ss.switchToSelectiveSearchFast() 
rects = ss.process()

print(f"Total region proposal: {len(rects)}")

# 3. Proses Deteksi & Filtrasi
boxes = []
for i, (x, y, w, h) in enumerate(rects[:2500]):
    
    # Filter ukuran objek
    if w < 30 or h < 30 or w > 300 or h > 300:
        continue
        
    # Filter Rasio Aspek (Lebar vs Tinggi) untuk meloloskan motor (ramping) dan mobil (lebar)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.35 or aspect_ratio > 2.0:
        continue
        
    # Potong sub-citra kandidat langsung dari gambar asli (bukan gambar bermasker) 
    # agar SVM mendapat tekstur asli kendaraan tanpa potongan hitam masker
    global_x_check = x + xmin
    global_y_check = y + ymin
    
    roi = img[global_y_check:global_y_check+h, global_x_check:global_x_check+w]
    if roi.size == 0:
        continue
        
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, IMG_SIZE)
    
    # Ekstraksi fitur HOG (8100 dimensi)
    fd = hog(roi_resized, orientations=9, pixels_per_cell=(4, 4), 
             cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    # Klasifikasi
    confidence = svm_model.decision_function([fd])[0]
    prediction = svm_model.predict([fd])[0]
    
    if prediction == 1 and confidence > CONFIDENCE_THRESHOLD:
        boxes.append([global_x_check, global_y_check, global_x_check + w, global_y_check + h])

# 4. Non-Maximum Suppression (NMS)
boxes = np.array(boxes)
if len(boxes) > 0:
    picked_boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.18)
else:
    picked_boxes = []

total_kendaraan = len(picked_boxes)

# 5. Visualisasi Bounding Box & Garis Batas Koridor
for (startX, startY, endX, endY) in picked_boxes:
    cv2.rectangle(img_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Tampilkan garis batas koridor jalan (Opsional, sangat bagus untuk visualisasi demo)
cv2.polylines(img_copy, [pts], True, (255, 0, 0), 2)

# 6. Logika Durasi Lampu Lalu Lintas [cite: 138-142]
durasi_hijau = 30
tambahan_waktu = 0

if total_kendaraan <= 5:
    tambahan_waktu = 0
elif 6 <= total_kendaraan <= 10:
    tambahan_waktu = 10
elif total_kendaraan > 10:
    tambahan_waktu = 15

durasi_total = durasi_hijau + tambahan_waktu

print("\n=== HASIL ANALISIS OPTIMAL ===")
print(f"Total Kendaraan Terdeteksi : {total_kendaraan}")
print(f"Rekomendasi Durasi Hijau   : {durasi_total} detik")

cv2.putText(img_copy, f'Kendaraan: {total_kendaraan}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(img_copy, f'Durasi Hijau: {durasi_total}s', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Hasil Deteksi Lampu Lalu Lintas - Koridor Jalan", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()