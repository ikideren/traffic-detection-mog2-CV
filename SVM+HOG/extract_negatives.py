import cv2
import os

# Konfigurasi path
TEST_IMAGE = 'test_intersection.jpg'
OUTPUT_DIR = 'dataset/train1/background'

img = cv2.imread(TEST_IMAGE)
if img is None:
    raise FileNotFoundError("Gambar test_intersection.jpg tidak ditemukan.")

# Koordinat kotak False Positive dari gambar Anda (Marka jalan dan aspal kosong)
# Format: [xmin, ymin, xmax, ymax] berdasarkan estimasi visual hasil deteksi liar Anda
false_positives = [
    [585, 675, 625, 990],   # Marka jalan vertikal di bawah
    [560, 350, 800, 680],   # Area aspal kosong di lajur kanan (tengah)
    [875, 425, 960, 485],   # Kotak kecil aspal di kanan tengah
    [780, 270, 995, 395],   # Area bayangan/aspal kanan atas
    [0, 275, 190, 710]      # Sektor pembatas jalan/tanaman kiri bawah
]

print("Mengekstrak sampel negatif baru...")
existing_count = len(os.listdir(OUTPUT_DIR))

for i, box in enumerate(false_positives):
    xmin, ymin, xmax, ymax = box
    # Potong area yang salah terdeteksi
    crop = img[ymin:ymax, xmin:xmax]
    
    if crop.size > 0:
        # Simpan ke folder background dengan indeks unik lanjutan
        file_name = f"hard_neg_{existing_count + i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), crop)
        print(f"Tersimpan: {file_name}")

print("Ekstraksi selesai. Silakan jalankan kembali preprocess.py dan train_svm.py!")