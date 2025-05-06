# -*- coding: utf-8 -*-
"""
Created on Mon May  5 22:44:13 2025

@author: Bagirathan
"""

import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO

# Load model
model = YOLO("E:/Senthil/AI/6.Deep Learning/Object detector with Voice/runs/detect/object_detector_yolov8/weights/best.pt")
print("[INFO] Model loaded successfully.")

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

# Threaded speech
def speak(text):
    def talk():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=talk, daemon=True).start()

# Speak once
speak("System check complete. Object detection is live.")

# Detection config
cooldown = 5  # seconds
conf_threshold = 0.1
MIN_HEIGHT = 80
important_classes = set(model.names.values())

# Track last seen + last spoken
last_spoken = {}
visible_objects = {}

print("[INFO] Detection started... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    results = model(frame, verbose=False)[0]
    detected_now = {}

    if len(results.boxes) == 0:
        print("[DEBUG] No detections.")

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if label not in important_classes or conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        height = y2 - y1

        if height < MIN_HEIGHT:
            continue

        # Track visible objects
        detected_now[label] = detected_now.get(label, 0) + 1

        center_x = (x1 + x2) // 2
        direction = (
            "to your left" if center_x < frame_width * 0.30 else
            "to your right" if center_x > frame_width * 0.60 else
            "ahead"
        )

        distance = (
            "very close" if height > 250 else
            "close" if height > 150 else
            "far"
        )

        spoken_text = f"{label} {direction}, {distance}"
        now = time.time()

        # Speak only if:
        # 1. It's newly detected
        # 2. Not visible in last frame
        # 3. Enough time passed since last announcement
        if (label not in visible_objects) and \
           (spoken_text not in last_spoken or now - last_spoken[spoken_text] > cooldown):
            print(f"[VOICE] Saying: {spoken_text}")
            speak(spoken_text)
            last_spoken[spoken_text] = now

        # Draw on screen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, f"{label}", (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Update visible objects
    visible_objects = detected_now.copy()

    cv2.imshow("Object Detector with Voice", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[INFO] 'Q' pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("[✅] Live detection session ended.")
