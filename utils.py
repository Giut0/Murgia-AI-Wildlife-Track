import re
import cv2 as cv
from PIL import Image

def estrai_data_ora(ocr_text):
    # Pulizia spazi anomali tipo "20:14 :19" → "20:14:19"
    ocr_text = re.sub(r'(\d{2}:\d{2})\s?:\s?(\d{2})', r'\1:\2', ocr_text)

    # Possibili pattern combinati (data + ora in una stringa)
    date_time_patterns = [
        # Orario PRIMA della data
        r"\b(?P<ora>\d{2}[:\-]\d{2}[:\-]\d{2})[ ]+(?P<data>\d{2}[\/\-\.]\d{1,3}[\/\-\.]\d{2,4})\b",
        r"\b(?P<ora>\d{2}[:\-]\d{2}[:\-]\d{2})[ ]+(?P<data>\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b",

        # Data PRIMA dell'orario
        r"\b(?P<data>\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})[ ]+(?P<ora>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
        r"\b(?P<data>\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})[ ]+(?P<ora>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
        r"\b(?P<data>\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2})[ ]+(?P<ora>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
    ]

    for pattern in date_time_patterns:
        match = re.search(pattern, ocr_text)
        if match:
            return match.group('data'), match.group('ora')

    return None, None  # Se nessun match

def get_best_frame(video, detection_model, conf_threshold, frame_interval):
    # Function to get the detected frame
    best_conf = 0.0
    best_frame = None
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue  # Skip frames

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
                best_bounding_box = [d['bbox'] for d in detections_above_threshold if d['conf'] == max_conf][0]

                if best_conf >= conf_threshold:
                    break 

    video.release()
    return best_frame, best_conf, best_bounding_box