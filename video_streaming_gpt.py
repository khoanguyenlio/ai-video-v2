import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
from datetime import datetime
import sys


def read_stream_with_pose_and_hands(stream_url, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                                    detection_confidence=0.5, tracking_confidence=0.5):
    print(f"Using OpenCV version: {cv2.__version__}")

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        smooth_landmarks=smooth_landmarks
    )

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    )

    print(f"Attempting to connect to stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Failed to open stream {stream_url}.")
        return

    print(f"Stream connected: {stream_url}")
    print(f"Frame width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"Frame height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    frame_count = 0
    all_data = []
    processing_times = []
    retry_count = 0
    max_retries = 3

    while True:
        ret, frame = cap.read()

        if not ret:
            retry_count += 1
            print(f"Lost connection. Retry {retry_count}/{max_retries}")
            if retry_count > max_retries:
                print("Maximum retries reached. Exiting.")
                break
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print("Failed to reconnect")
                break
            continue

        retry_count = 0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_data = {"frame": frame_count}

        if frame_count % process_every_n == 0:
            start_time = datetime.now()

            pose_results = pose.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            landmarks_list = []
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    landmarks_list.append({
                        "type": "pose",
                        "index": idx,
                        "name": mp_pose.PoseLandmark(idx).name,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

            hand_list = []
            if hand_results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    label = hand_results.multi_handedness[i].classification[0].label
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        hand_list.append({
                            "type": "hand",
                            "hand": label,
                            "index": idx,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            processing_times.append(processing_time)

            frame_data["pose_landmarks"] = landmarks_list
            frame_data["hand_landmarks"] = hand_list
            frame_data["processing_time_ms"] = processing_time
            all_data.append(frame_data)

            cv2.putText(frame, f"{processing_time:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Pose and Hand Detection', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"stream_pose_hand_landmarks_{timestamp}.json")

    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    try:
        with open(log_file, 'w') as f:
            json.dump({
                "stream_url": stream_url,
                "total_frames": frame_count,
                "processed_frames": len(all_data),
                "avg_processing_time_ms": avg_processing_time,
                "data": all_data
            }, f, indent=2)
        print(f"Landmark data saved to: {log_file}")
    except Exception as e:
        print(f"Error saving landmark data: {e}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read RTSP stream and detect human pose & hands')
    parser.add_argument('--stream_url', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save landmark logs')
    parser.add_argument('--no_smooth', action='store_false', dest='smooth', help='Disable landmark smoothing')
    parser.add_argument('--process_rate', type=int, default=3, help='Process every n frames')
    parser.add_argument('--detection_conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--tracking_conf', type=float, default=0.5, help='Tracking confidence threshold')
    args = parser.parse_args()

    read_stream_with_pose_and_hands(
        args.stream_url,
        args.log_dir,
        args.smooth,
        args.process_rate,
        args.detection_conf,
        args.tracking_conf
    )
