import numpy as np
import cv2 as cv
from PIL import Image
import torch
from megadetector.detection import run_detector
from megadetector.visualization import visualization_utils as vis_utils
import easyocr
import re
from utils import *
from utils import get_best_frame
reader = easyocr.Reader(['en', 'it'])  # 'en' per date, 'it' per °C

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model
detection_model = run_detector.load_detector("MDV5A", force_cpu=(device == "cpu"))

# Load video
cap = cv.VideoCapture('data/labeled_videos/gazza 2bis (1).AVI')



CONFIDENCE_THRESHOLD_EARLY_STOP = 0.80

frame_interval = 30  # Analyze every 30th frame




best_frame, best_conf, best_bounding_box = get_best_frame(cap, detection_model ,CONFIDENCE_THRESHOLD_EARLY_STOP, frame_interval)

print(f"Confidenza massima trovata: {best_conf:.3f}")
print(f"Bounding box: {best_bounding_box}")

if best_frame is not None:
    cv.imshow('Frame con confidenza massima', best_frame)
    results = reader.readtext(best_frame)


    # Estrae solo il testo da ogni blocco e lo unisce
    full_text = ' '.join([text for _, text, _ in results])

    data_str, ora_str = estrai_data_ora(full_text)
    print("📅 Data trovata:", data_str)
    print("🕒 Ora trovata:", ora_str)

    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Nessuna detection trovata nel video.")