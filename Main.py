import sys
import struct

print("Python Version:", sys.version)
print("Bit Architecture:", struct.calcsize("P") * 8, "-bit")

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import csv
import os
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH = 'emotion_model.h5'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SMOOTHING_WINDOW = 10          # frames to average predictions over
CONFIDENCE_THRESHOLD = 0.40   # only display if confidence >= this
LOG_FILE = 'emotion_log.csv'
BAR_WIDTH = 200                # width of probability bar panel

EMOTION_COLORS = {
    'Anger':    (0,   0,   255),
    'Disgust':  (0,   140, 255),
    'Fear':     (0,   255, 255),
    'Happy':    (0,   255, 0),
    'Sad':      (255, 0,   0),
    'Surprise': (255, 0,   255),
    'Neutral':  (200, 200, 200),
}

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
print("[INFO] Loading model...")
model = load_model(MODEL_PATH, compile=False)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("[INFO] Model loaded.")

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# ─────────────────────────────────────────
# FEATURE 2: Emotion Smoothing Buffer
# One deque per emotion class, stores recent probabilities
# ─────────────────────────────────────────
emotion_history = deque(maxlen=SMOOTHING_WINDOW)

# ─────────────────────────────────────────
# FEATURE 5: CSV Logger
# ─────────────────────────────────────────
log_exists = os.path.exists(LOG_FILE)
log_file = open(LOG_FILE, 'a', newline='')
csv_writer = csv.writer(log_file)
if not log_exists:
    csv_writer.writerow(['timestamp', 'emotion', 'confidence'] + EMOTION_LABELS)


def log_emotion(emotion, confidence, probs):
    """Write one row to the CSV log."""
    row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), emotion, f"{confidence:.4f}"]
    row += [f"{p:.4f}" for p in probs]
    csv_writer.writerow(row)


# ─────────────────────────────────────────
# FEATURE 4: Draw Probability Bar Panel
# ─────────────────────────────────────────
def draw_bar_panel(frame, smoothed_probs):
    """
    Draws a vertical panel on the right side of the frame
    showing a probability bar for each emotion.
    """
    h, w = frame.shape[:2]
    panel = np.zeros((h, BAR_WIDTH, 3), dtype=np.uint8)
    bar_area_height = h - 20
    bar_h = bar_area_height // len(EMOTION_LABELS)

    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, smoothed_probs)):
        y_start = 10 + i * bar_h
        color = EMOTION_COLORS[label]
        filled_w = int(prob * (BAR_WIDTH - 10))

        # Background bar
        cv2.rectangle(panel, (5, y_start + 2), (BAR_WIDTH - 5, y_start + bar_h - 4),
                      (50, 50, 50), -1)
        # Filled portion
        if filled_w > 0:
            cv2.rectangle(panel, (5, y_start + 2), (5 + filled_w, y_start + bar_h - 4),
                          color, -1)

        # Label and percentage
        text = f"{label[:3]} {prob*100:.0f}%"
        cv2.putText(panel, text, (7, y_start + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    combined = np.hstack([frame, panel])
    return combined


# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# FEATURE 3: FPS tracking
fps_start = time.time()
fps_counter = 0
fps_display = 0.0

print("[INFO] Running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame read failed.")
        break

    # ── FPS counter (Feature 3) ──────────────────────────
    fps_counter += 1
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        fps_display = fps_counter / elapsed
        fps_counter = 0
        fps_start = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # Default: if no face, push neutral uniform distribution to keep history valid
    current_probs = np.ones(len(EMOTION_LABELS)) / len(EMOTION_LABELS)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64)).astype('float32') / 255.0
        face_input = face_resized[np.newaxis, :, :, np.newaxis]   # (1,64,64,1)

        # Predict
        raw_probs = model.predict(face_input, verbose=0)[0]        # shape (7,)
        current_probs = raw_probs

        # ── Feature 2: Smooth over history ────────────────
        emotion_history.append(raw_probs)
        smoothed_probs = np.mean(emotion_history, axis=0)

        # ── Feature 1: Confidence score ───────────────────
        best_idx = int(np.argmax(smoothed_probs))
        best_label = EMOTION_LABELS[best_idx]
        confidence = float(smoothed_probs[best_idx])

        # Draw face box
        box_color = EMOTION_COLORS[best_label]
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        if confidence >= CONFIDENCE_THRESHOLD:
            label_text = f"{best_label}  {confidence*100:.1f}%"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)
        else:
            cv2.putText(frame, "Uncertain", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

        # ── Feature 5: Log every 30 frames ────────────────
        if fps_counter % 30 == 0:
            log_emotion(best_label, confidence, smoothed_probs)

        # Only process first detected face for clarity
        break

    # ── Feature 4: Probability bar panel ──────────────────
    smoothed = (np.mean(emotion_history, axis=0)
                if len(emotion_history) > 0
                else current_probs)
    display_frame = draw_bar_panel(frame, smoothed)

    # ── Feature 3: FPS overlay ────────────────────────────
    cv2.putText(display_frame, f"FPS: {fps_display:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Faces count
    cv2.putText(display_frame, f"Faces: {len(faces)}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Emotion Detection — Enhanced", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
log_file.close()
print(f"[INFO] Session log saved to '{LOG_FILE}'")
