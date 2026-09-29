import cv2
import numpy as np
import math

VIDEO_PATH = 'MOG/traffic_video6.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Error: Video '{VIDEO_PATH}' tidak dapat diakses.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# -------------------------------------------------------------
# INISIALISASI 4 EXPORT VIDEO WRITER TERPISAH
# -------------------------------------------------------------
out_layer1 = cv2.VideoWriter('MOG/layer1_roi.mp4', fourcc, fps, (frame_width, frame_height))
out_layer2 = cv2.VideoWriter('MOG/layer2_raw_mog.mp4', fourcc, fps, (frame_width, frame_height))
out_layer3 = cv2.VideoWriter('MOG/layer3_morphological.mp4', fourcc, fps, (frame_width, frame_height))
out_layer4 = cv2.VideoWriter('MOG/layer4_final_tracking.mp4', fourcc, fps, (frame_width, frame_height))

mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

tracked_vehicles = {}
next_vehicle_id = 0
MAX_STATIONARY_FRAMES = 900

print("Memulai pemrosesan video dan mengekspor 4 file mp4...")

while True:
    ret, frame = cap.read()
    if not ret:
         break

    h, w = frame.shape[:2]
    
    
    # LAYER 1: ORIGINAL + DYNAMIC ROI MASKING
    frame_roi_viz = frame.copy()
    roi_y1, roi_y2 = int(h * 0.2), h
    roi_x1, roi_x2 = int(w * 0.15), int(w * 0.75)
    
    cv2.rectangle(frame_roi_viz, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)
    cv2.putText(frame_roi_viz, "Layer 1: Dynamic ROI", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]


    # LAYER 2: RAW MOG2 BACKGROUND SUBTRACTION
    raw_fg_mask = mog2.apply(roi, learningRate=0.005)
    _, raw_fg_mask = cv2.threshold(raw_fg_mask, 254, 255, cv2.THRESH_BINARY)
    
    full_raw_mask = np.zeros((h, w), dtype=np.uint8)
    full_raw_mask[roi_y1:roi_y2, roi_x1:roi_x2] = raw_fg_mask
    
    # Konversi 1-channel ke 3-channel (BGR) untuk export video
    layer2_bgr = cv2.cvtColor(full_raw_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(layer2_bgr, "Layer 2: Raw MOG2", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # LAYER 3: MORPHOLOGICAL FILTERING (CLEANED MASK)
    fg_mask = cv2.morphologyEx(raw_fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    full_filtered_mask = np.zeros((h, w), dtype=np.uint8)
    full_filtered_mask[roi_y1:roi_y2, roi_x1:roi_x2] = fg_mask
    
    # Konversi 1-channel ke 3-channel (BGR) untuk export video
    layer3_bgr = cv2.cvtColor(full_filtered_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(layer3_bgr, "Layer 3: Morphological Filter", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


    # LAYER 4: CENTROID TRACKING & DYNAMIC QUEUING LOGIC
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_frame_centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = float(w_box) / h_box
            if 0.3 < aspect_ratio < 3.0:
                global_x = x + roi_x1
                global_y = y + roi_y1
                cX = global_x + w_box // 2
                cY = global_y + h_box // 2
                current_frame_centroids.append((cX, cY, global_x, global_y, w_box, h_box))

    updated_vehicles = {}
    for (cX, cY, gx, gy, gw, gh) in current_frame_centroids:
        matched = False
        best_match_id = None
        min_dist = 50
        
        for v_id, (old_x, old_y, old_w, old_h, age) in tracked_vehicles.items():
            old_cX = old_x + old_w // 2
            old_cY = old_y + old_h // 2
            distance = math.sqrt((cX - old_cX)**2 + (cY - old_cY)**2)
            if distance < min_dist:
                min_dist = distance
                best_match_id = v_id
                
        if best_match_id is not None:
            updated_vehicles[best_match_id] = (gx, gy, gw, gh, 0)
            matched = True
        if not matched:
            updated_vehicles[next_vehicle_id] = (gx, gy, gw, gh, 0)
            next_vehicle_id += 1

    for v_id, (old_x, old_y, old_w, old_h, age) in tracked_vehicles.items():
        if v_id not in updated_vehicles:
            margin = 15
            if (old_y < roi_y1 + margin) or ((old_y + old_h) > roi_y2 - margin) or (old_x < roi_x1 + margin) or ((old_x + old_w) > roi_x2 - margin):
                continue
            if age < MAX_STATIONARY_FRAMES:
                updated_vehicles[v_id] = (old_x, old_y, old_w, old_h, age + 1)

    tracked_vehicles = updated_vehicles.copy()
    total_kendaraan = next_vehicle_id

    for v_id, (x, y, w_box, h_box, age) in tracked_vehicles.items():
        color = (0, 0, 255) if age > 5 else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
        cv2.putText(frame, f"ID:{v_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    T_base, t_clear, T_max = 15, 2.5, 60
    kalkulasi_waktu = T_base + (total_kendaraan * t_clear)
    durasi_total = min(T_max, int(kalkulasi_waktu))

    cv2.putText(frame, f'Total Mobil: {total_kendaraan}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame, f'Waktu Hijau: {durasi_total}s', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame, "Layer 4: Tracking & Logic", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # -------------------------------------------------------------
    # PENULISAN MASING-MASING FRAME KE FILE MP4
    # -------------------------------------------------------------
    out_layer1.write(frame_roi_viz)
    out_layer2.write(layer2_bgr)
    out_layer3.write(layer3_bgr)
    out_layer4.write(frame)

    # Menampilkan window
    cv2.imshow("Layer 1 - Dynamic ROI", frame_roi_viz)
    cv2.imshow("Layer 2 - Raw MOG2", layer2_bgr)
    cv2.imshow("Layer 3 - Morphological Filter", layer3_bgr)
    cv2.imshow("Layer 4 - Tracking & Decision Logic", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out_layer1.release()
out_layer2.release()
out_layer3.release()
out_layer4.release()
cv2.destroyAllWindows()

print("Proses Selesai! 4 File MP4 telah berhasil diekspor di folder MOG/.")