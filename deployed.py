import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("E:/Senthil/AI/6.Deep Learning/Object detector with Voice/runs/detect/train/weights/best.pt")
print("[INFO] Model loaded successfully.")

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

# Initialize TTS engine globally (to avoid RuntimeError)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Threaded speech function
def speak(text):
    def talk():
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            pass  # Prevent 'run loop already started' errors
    threading.Thread(target=talk, daemon=True).start()

# Speak once at startup
speak("System check complete. Object detection is live.")

# Detection settings
cooldown = 5  # seconds to prevent repeated alerts
conf_threshold = 0.7  # stricter detection threshold
MIN_HEIGHT = 80  # pixel height to ignore far objects
important_classes = set(model.names.values())  # track all classes

# Track recent alerts and visible objects
last_spoken = {}
visible_objects = {}

print("[INFO] Detection started... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror view (like a selfie camera)
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

        print(f"[DEBUG] Detected: {label} ({conf:.2f}) at ({x1}, {y1}), {direction}, {distance}")

        if (label not in visible_objects) and \
           (spoken_text not in last_spoken or now - last_spoken[spoken_text] > cooldown):
            print(f"[VOICE] Saying: {spoken_text}")
            speak(spoken_text)
            last_spoken[spoken_text] = now

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, f"{label}", (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Track which objects are visible
    visible_objects = detected_now.copy()

    # Display annotated frame
    cv2.imshow("Object Detector with Voice", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[INFO] 'Q' pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("[✅] Live detection session ended.")