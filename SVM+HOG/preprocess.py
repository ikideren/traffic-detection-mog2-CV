import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from tqdm import tqdm # Untuk menampilkan progress bar

# 1. Setup path dan kelas
DATASET_PATH = 'dataset/train1'
POSITIVE_CLASSES = ['car', 'bus', 'motorcycle', 'pickup_truck', 'articulated_truck', 'single_unit_truck', 'work_van']
NEGATIVE_CLASSES = ['background']
IMG_SIZE = (64, 64)

features = []
labels = []

# 2. Fungsi untuk ekstraksi
def extract_hog_features(image_path, label):
    # Baca gambar dalam grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Resize gambar
        img_resized = cv2.resize(img, IMG_SIZE)
        # Ekstraksi HOG
        fd = hog(img_resized, orientations=9, pixels_per_cell=(4, 4), 
                 cells_per_block=(2, 2), visualize=False)
        features.append(fd)
        labels.append(label)

# 3. Looping kelas positif (Label 1)
print("Mengekstrak kelas positif...")
for cls in POSITIVE_CLASSES:
    folder_path = os.path.join(DATASET_PATH, cls)
    for img_name in tqdm(os.listdir(folder_path)):
        extract_hog_features(os.path.join(folder_path, img_name), 1)

# 4. Looping kelas negatif (Label 0)
print("Mengekstrak kelas negatif...")
for cls in NEGATIVE_CLASSES:
    folder_path = os.path.join(DATASET_PATH, cls)
    for img_name in tqdm(os.listdir(folder_path)):
        extract_hog_features(os.path.join(folder_path, img_name), 0)

# 5. Konversi ke Numpy array dan Simpan
X = np.array(features)
y = np.array(labels)

print(f"Total data: {len(X)}")
joblib.dump((X, y), 'hog_features_labels.pkl')
print("Fitur berhasil disimpan ke hog_features_labels.pkl!")