import numpy as np
import cv2 as cv
from PIL import Image
import torch
from megadetector.detection import run_detector
from megadetector.visualization import visualization_utils as vis_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model
detection_model = run_detector.load_detector("MDV5A", force_cpu=(device == "cpu"))

# Load video
cap = cv.VideoCapture('data/labeled_videos/gazza 2bis (1).AVI')


best_conf = 0.0
best_frame = None
best_detections = []
CONFIDENCE_THRESHOLD_EARLY_STOP = 0.80

frame_interval = 30  # Analyze every 30th frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue  # Skip frames

    # Resize the frame to 540x380
    frame = cv.resize(frame, (540, 380), interpolation=cv.INTER_CUBIC)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Apply detection
    result = detection_model.generate_detections_one_image(image)
    detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]

    # Choose the best detection
    if detections_above_threshold:
        max_conf = max(d['conf'] for d in detections_above_threshold)
        if max_conf > best_conf:
            best_conf = max_conf
            best_frame = frame.copy()
            best_detections = detections_above_threshold.copy()

            if best_conf >= CONFIDENCE_THRESHOLD_EARLY_STOP:
                print(f"Confidenza alta trovata: {best_conf:.3f}, fermo l'elaborazione.")
                break  # Fermati subito

cap.release()

for i, detection in enumerate(best_detections):
    print(f"Detection {i}: {detection}")
    print(f"Confidenza: {detection['conf']:.3f}")
    print(f"Coordinate: {detection['bbox']}")

if best_frame is not None:
    cv.imshow('Frame con confidenza massima', best_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Nessuna detection trovata nel video.")