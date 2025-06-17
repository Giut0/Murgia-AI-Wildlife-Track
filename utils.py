import os
import re
import cv2 as cv
from PIL import Image
from dateutil import parser

def normalize_datetime(date_str):
    try:
        dt = parser.parse(date_str, dayfirst=True)  
        return dt.isoformat()
    except Exception as e:
        return None
    
def get_unique_filename(path):
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{base} ({counter}){ext}"
    
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base} ({counter}){ext}"

    return new_path

def translate_filename(filename, trans_dic):

    filename = filename.lower()

    for eng_label, italian_words in trans_dic.items():
        for italian_word in italian_words:
            if italian_word in filename:
                return eng_label

    return "unknown"

def extract_date_time(ocr_text):
    # Clean up the OCR text
    ocr_text = re.sub(r'(\d{2}:\d{2})\s?:\s?(\d{2})', r'\1:\2', ocr_text)

    # Types of date and time formats
    date_time_patterns = [
        r"\b(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})[ ]+(?P<date>\d{2}[\/\-\.]\d{1,3}[\/\-\.]\d{2,4})\b",
        r"\b(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})[ ]+(?P<date>\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b",
        r"\b(?P<date>\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})[ ]+(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
        r"\b(?P<date>\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})[ ]+(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
        r"\b(?P<date>\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2})[ ]+(?P<time>\d{2}[:\-]\d{2}[:\-]\d{2})\b",
    ]

    for pattern in date_time_patterns:
        match = re.search(pattern, ocr_text)
        if match:
            return match.group('date'), match.group('time')

    return None, None

def get_best_frame(video, detection_model, conf_threshold, frame_interval):
    best_conf = 0.0
    best_frame = None
    frame_count = 0
    best_bounding_box = None
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