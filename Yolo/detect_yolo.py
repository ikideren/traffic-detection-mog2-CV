import cv2
from ultralytics import YOLO

# 1. Muat model YOLOv8 versi 'nano' (sangat ringan dan cepat)
# Model akan otomatis terunduh (sekitar 6MB) saat pertama kali dijalankan
model = YOLO('yolov8n.pt')

# 2. Baca citra uji
TEST_IMAGE = 'test_intersection.jpg'
img = cv2.imread(TEST_IMAGE)

if img is None:
    raise FileNotFoundError(f"Error: Gambar '{TEST_IMAGE}' tidak ditemukan.")

# 3. Proses Deteksi
# classes=[2, 3, 5, 7] memfilter agar hanya mendeteksi: mobil, motor, bus, dan truk
# conf=0.3 adalah batas keyakinan yang sangat aman untuk YOLO
results = model(img, classes=[2, 3, 5, 7], conf=0.3)

# 4. Ambil hasil deteksi (Bounding boxes dan total kendaraan)
# YOLO otomatis menangani Non-Maximum Suppression (NMS) di latar belakang
boxes = results[0].boxes
total_kendaraan = len(boxes)

# Gambar bounding box bawaan YOLO yang sangat rapi ke atas citra
img_annotated = results[0].plot()

# 5. Logika Durasi Lampu Lalu Lintas
durasi_hijau = 30
tambahan_waktu = 0

if total_kendaraan <= 5:
    tambahan_waktu = 0
elif 6 <= total_kendaraan <= 10:
    tambahan_waktu = 10
elif total_kendaraan > 10:
    tambahan_waktu = 15

durasi_total = durasi_hijau + tambahan_waktu

# 6. Tampilkan Hasil
print("\n=== HASIL ANALISIS YOLOv8 ===")
print(f"Total Kendaraan Terdeteksi : {total_kendaraan}")
print(f"Rekomendasi Durasi Hijau   : {durasi_total} detik")

cv2.putText(img_annotated, f'Kendaraan: {total_kendaraan}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(img_annotated, f'Durasi Hijau: {durasi_total}s', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Optimal Detection - YOLOv8", img_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()