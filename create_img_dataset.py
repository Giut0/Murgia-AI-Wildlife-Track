import os
import csv
import torch
import easyocr
import cv2 as cv
from utils import *
import pandas as pd
from megadetector.detection import run_detector

trans_dic = {
    "fox" : ["volpe"],
    "wolf" : ["lupo"],
    "cat" : ["gatto"],
    "badger" : ["tasso"],
    "weasel" : ["faina"],
    "dog" : ["cane"],
    "bird" : ["cinciallegra", "gazza", "cinciarella", "verzellino", "upupa", "ghiandaia", "pigliamosche", "pettirosso", "passero"],
    "lizard" : ["lucertola"],
    "snake" : ["serpente"],
    "bug" : ["coleottero"],
    "butterfly" : ["farfalla"],
    "boar" : ["cinghiale"],
    "podolic_cow" : ["vacca podolica", "podolica"],
    "porcupine" : ["istrice"],
    "mouse" : ["topo", "ratto"]
}

# Load model detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
detection_model = run_detector.load_detector("MDV5A", force_cpu=(device == "cpu"))

timestamp = ""
CONFIDENCE_THRESHOLD_EARLY_STOP = 0.80

frame_interval = 5  # Analyze every 5th frame

folder_path = 'data/labeled_videos'
output_csv = 'data/labeled_img.csv'
iid = 671
with open(output_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filepath', 'class', 'timestamp', 'x_min', 'y_min', 'width', 'height'])
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            cap = cv.VideoCapture(file_path)
            # Apply the detection model to the video to get the best frame
            best_frame, best_conf, best_bounding_box = get_best_frame(cap, detection_model ,CONFIDENCE_THRESHOLD_EARLY_STOP, frame_interval)

            if best_frame is not None:
                # Add translated filename to the detected frame
                label = translate_filename(os.path.basename(file_path), trans_dic)
                if label != "unknown":
                    # Save the best frame with the label
                    safe_filename = get_unique_filename("data/labeled_img/"+label+".png")
                    cv.imwrite(safe_filename, best_frame)
                    # Extract date and time from the best frame using EasyOCR
                    reader = easyocr.Reader(['en', 'it']) 
                    results = reader.readtext(best_frame)
                    full_text = ' '.join([text for _, text, _ in results])
                    data_str, ora_str = extract_date_time(full_text)

                    if data_str is None or ora_str is None:
                            timestamp = pd.NaT
                    else:
                        timestamp = normalize_datetime(data_str + " " + ora_str)
                    # Write the data to the CSV file
                    writer.writerow([label+".png", label, timestamp, best_bounding_box[0], best_bounding_box[1], best_bounding_box[2], best_bounding_box[3]])
            else:
                pass