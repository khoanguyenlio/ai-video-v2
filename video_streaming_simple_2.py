import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
from datetime import datetime, timedelta
import sys
import math
from ultralytics import YOLO

def is_significant_movement(prev_landmarks, curr_landmarks, threshold=0.05):
    if prev_landmarks is None or curr_landmarks is None:
        return False
    total_movement = 0
    count = 0
    for p, c in zip(prev_landmarks, curr_landmarks):
        dx = abs(p['x'] - c['x'])
        dy = abs(p['y'] - c['y'])
        dz = abs(p['z'] - c['z'])
        total_movement += math.sqrt(dx*dx + dy*dy + dz*dz)
        count += 1
    avg_movement = total_movement / count if count > 0 else 0
    return avg_movement > threshold

def apply_landmark_smoothing(current_landmarks, landmark_history, smoothing_factor=0.5):
    """Apply temporal smoothing to landmarks using exponential moving average."""
    if not landmark_history:
        return current_landmarks
    
    smoothed_landmarks = []
    for i, curr_lm in enumerate(current_landmarks):
        x_sum = curr_lm['x']
        y_sum = curr_lm['y']
        z_sum = curr_lm['z']
        count = 1
        
        # Average with historical landmarks
        for history in landmark_history:
            if i < len(history):
                hist_lm = history[i]
                x_sum += hist_lm['x'] * smoothing_factor
                y_sum += hist_lm['y'] * smoothing_factor
                z_sum += hist_lm['z'] * smoothing_factor
                count += smoothing_factor
        
        # Create new smoothed landmark
        smoothed_landmarks.append({
            "index": curr_lm["index"],
            "name": curr_lm["name"],
            "x": x_sum / count,
            "y": y_sum / count,
            "z": z_sum / count,
            "visibility": curr_lm["visibility"]
        })
    
    return smoothed_landmarks

def read_video_with_pose(video_path, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                         detection_confidence=0.5, tracking_confidence=0.5,
                         draw_yolo_box=False, draw_pose_box=False,
                         max_frames_per_second=3):
    print(f"Using OpenCV version: {cv2.__version__}")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    yolo_model = YOLO("yolov8n.pt")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        smooth_landmarks=smooth_landmarks
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video stream {video_path}.")
        return

    # Get video FPS to calculate frame timing
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30fps if unable to determine
    frame_time_ms = 1000.0 / fps
    print(f"Video stream opened: {video_path} (FPS: {fps:.2f})")
    
    all_landmarks = []
    last_landmarks = None
    landmark_history = []
    max_history = 5
    consecutive_failures = 0
    processing_times = []
    frame_count = 0
    
    # Variables to track frame logging frequency
    last_log_time = None
    frames_logged_in_current_second = 0
    current_second = None

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate current timestamp based on frame count and FPS
            current_frame_time = timedelta(seconds=frame_count/fps)
            current_second_of_video = int(current_frame_time.total_seconds())
            
            # Reset counter when moving to a new second
            if current_second != current_second_of_video:
                current_second = current_second_of_video
                frames_logged_in_current_second = 0
            
            process_this_frame = frame_count % process_every_n == 0

            if process_this_frame:
                start_time = datetime.now()

                yolo_results = yolo_model(frame, verbose=False)[0]
                person_boxes = [box for box in yolo_results.boxes.data.tolist() if int(box[5]) == 0]

                for box in person_boxes:
                    x1, y1, x2, y2, conf, cls = box
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    if draw_yolo_box:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue

                    frame_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        consecutive_failures = 0
                        if len(landmark_history) >= max_history:
                            landmark_history.pop(0)
                        landmark_history.append(results.pose_landmarks)
                        last_landmarks = results.pose_landmarks

                        landmarks_list = []
                        for idx, landmark in enumerate(last_landmarks.landmark):
                            landmarks_list.append({
                                "index": idx,
                                "name": mp_pose.PoseLandmark(idx).name,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": landmark.visibility
                            })

                        mp_drawing.draw_landmarks(
                            frame[y1:y2, x1:x2],
                            last_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )

                        if draw_pose_box:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                        prev_frame_landmarks = all_landmarks[-1]["landmarks"] if all_landmarks else None
                        
                        # Check if we should log this frame
                        should_log = (frames_logged_in_current_second < max_frames_per_second and 
                                     is_significant_movement(prev_frame_landmarks, landmarks_list, threshold=0.035))
                        
                        if should_log:
                            frame_data = {
                                "frame": frame_count,
                                "time_seconds": current_second_of_video,
                                "bbox": [x1, y1, x2, y2],
                                "landmarks": landmarks_list,
                                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                            }
                            all_landmarks.append(frame_data)
                            frames_logged_in_current_second += 1
                            print(f"Logged frame {frame_count} (second {current_second_of_video}, frame {frames_logged_in_current_second}/{max_frames_per_second})")

                # Display info about logging status
                log_info = f"Processed: {frame_count} | Logged: {len(all_landmarks)} | Second: {current_second_of_video}"
                cv2.putText(
                    frame,
                    log_info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.imshow('Stream Pose Detection', frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            frame_count += 1
            consecutive_failures += 1
            if consecutive_failures > 10:
                print("Too many consecutive failures, stopping...")
                break

    print(f"Processing complete. Total frames: {frame_count}, Logged frames: {len(all_landmarks)}")
    
    if all_landmarks:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"stream_pose_landmarks_{timestamp}.json")

        avg_processing_time = sum([f["processing_time_ms"] for f in all_landmarks]) / len(all_landmarks)

        try:
            with open(log_file, 'w') as f:
                json.dump({
                    "video_path": video_path,
                    "total_frames": frame_count,
                    "processed_frames": len(all_landmarks),
                    "processing_rate": f"Every {process_every_n} frames",
                    "max_frames_per_second": max_frames_per_second,
                    "smoothing_enabled": smooth_landmarks,
                    "detection_confidence": detection_confidence,
                    "tracking_confidence": tracking_confidence,
                    "avg_processing_time_ms": avg_processing_time,
                    "landmarks": all_landmarks
                }, f, indent=2)
            print(f"Landmark data saved to: {log_file}")
            print(f"Average processing time: {avg_processing_time:.2f} ms per frame")
        except Exception as e:
            print(f"Error saving landmark data: {e}")
    else:
        print("No landmarks were detected and logged.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read stream and detect human pose')
    parser.add_argument('video_path', type=str, help='RTSP stream or video path')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save landmark logs')
    parser.add_argument('--no_smooth', action='store_false', dest='smooth', help='Disable landmark smoothing')
    parser.add_argument('--process_rate', type=int, default=1, help='Process every n frames')
    parser.add_argument('--detection_conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--tracking_conf', type=float, default=0.5, help='Tracking confidence threshold')
    parser.add_argument('--draw_yolo_box', action='store_true', help='Draw YOLOv8 person bounding boxes')
    parser.add_argument('--draw_pose_box', action='store_true', help='Draw MediaPipe pose bounding boxes')
    parser.add_argument('--max_frames_per_second', type=int, default=3, help='Maximum frames to log per second of video')
    args = parser.parse_args()

    read_video_with_pose(
        args.video_path,
        args.log_dir,
        args.smooth,
        args.process_rate,
        args.detection_conf,
        args.tracking_conf,
        draw_yolo_box=args.draw_yolo_box,
        draw_pose_box=args.draw_pose_box,
        max_frames_per_second=args.max_frames_per_second
    )