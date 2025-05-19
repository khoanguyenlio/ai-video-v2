import cv2
import numpy as np
import mediapipe as mp
import time
import os
from mediapipe.tasks.python.core import base_options as mp_base
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ẩn log TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ========= CONFIG ==========
MODEL_PATH = 'pose_landmarker_lite.task'
DEVICE_INDEX = 0
DRAW_KEYPOINTS = True
SMOOTHING_ENABLED = True
# ===========================

KEYPOINTS_TO_LOG = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle"
}

# Line connections
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Biến toàn cục
last_result = None
previous_result = None  # chỉ dùng cho smoothing

# Callback xử lý pose landmarks
def print_landmarks(result, output_image, timestamp):
    global last_result, previous_result
    try:
        if result.pose_landmarks:
            current_result = result.pose_landmarks[0]

            # Smoothing thủ công
            if SMOOTHING_ENABLED and previous_result and len(previous_result) == len(current_result):
                for i in range(len(current_result)):
                    current_result[i].x = (current_result[i].x + previous_result[i].x) / 2
                    current_result[i].y = (current_result[i].y + previous_result[i].y) / 2
                    current_result[i].z = (current_result[i].z + previous_result[i].z) / 2

            previous_result = current_result
            last_result = current_result

            print("🔹 Landmarks:")
            for idx, label in KEYPOINTS_TO_LOG.items():
                if idx < len(current_result):
                    lm = current_result[idx]
                    print(f"{label}: x={lm.x:.2f}, y={lm.y:.2f}, z={lm.z:.2f}, vis={lm.visibility:.2f}")
            print()
    except Exception as e:
        print(f"⚠️ Callback error: {e}")

# Init pose landmarker
options = vision.PoseLandmarkerOptions(
    base_options=mp_base.BaseOptions(model_asset_path=MODEL_PATH),
    output_segmentation_masks=True,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_landmarks
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam.")

start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp = int((time.time() - start_time) * 1e6)

        try:
            landmarker.detect_async(mp_image, timestamp)
        except Exception as e:
            print(f"⚠️ detect_async error: {e}")

        # Vẽ keypoints + lines
        if DRAW_KEYPOINTS and last_result:
            h, w, _ = frame.shape

            # Vẽ connections
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(last_result) and end_idx < len(last_result):
                    lm_start = last_result[start_idx]
                    lm_end = last_result[end_idx]
                    if lm_start.visibility > 0.6 and lm_end.visibility > 0.6:
                        x0, y0 = int(lm_start.x * w), int(lm_start.y * h)
                        x1, y1 = int(lm_end.x * w), int(lm_end.y * h)
                        cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)

            # Vẽ keypoints
            for idx, lm in enumerate(last_result):
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (0, 255, 0) if (idx in KEYPOINTS_TO_LOG and lm.visibility >= 0.9) else (0, 0, 255)
                cv2.circle(frame, (cx, cy), 5, color, -1)

        cv2.imshow("Pose Landmarker (Live)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()
