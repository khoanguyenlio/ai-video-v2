import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
from datetime import datetime
import sys
from ultralytics import YOLO  # Import YOLO

def read_video_with_pose(video_path, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                         detection_confidence=0.5, tracking_confidence=0.5,
                         yolo_conf=0.5, smoothing_factor=0.7):
    """
    Read a video file, detect people using YOLOv8, and then detect human pose using MediaPipe
    """
    # First verify the video file exists
    if not video_path.startswith("rtsp://") and not os.path.exists(video_path):
        print(f"Error: Video source '{video_path}' not found.")
        return
    
    print(f"Using OpenCV version: {cv2.__version__}")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize YOLO model
    try:
        yolo_model = YOLO("yolov8n.pt")
        print("YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Using 1 instead of 2 for better performance
        enable_segmentation=False,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        smooth_landmarks=smooth_landmarks
    )
    
    # Try opening the video file
    try:
        print(f"Attempting to open video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Failed to open video {video_path}. Check file format and codecs.")
            return
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    print(f"Video opened: {video_path}")
    print(f"Frame width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"Frame height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Main video processing code
    all_landmarks = []
    
    # For smoothing landmarks between frames
    landmark_history = []
    landmarks_history_raw = []
    last_landmarks = None
    max_history = 5
    
    # For stabilizing detection
    consecutive_failures = 0
    max_failures = 10
    
    # For measuring performance
    processing_times = []
    
    # Read and display video frames
    frame_count = 0
    
    # Create output directories
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Main processing loop
    while True:
        try:
            ret, frame = cap.read()
            
            # Break the loop if we've reached the end of the video
            if not ret:
                break
            
            # Process every n frames
            process_this_frame = frame_count % process_every_n == 0
            
            frame_data = {"frame": frame_count, "landmarks": None}
            
            if process_this_frame:
                start_time = datetime.now()
                
                # First use YOLO to detect people
                yolo_results = yolo_model(frame, verbose=False)[0]
                # Filter for person class (class 0 in COCO dataset)
                person_boxes = [box for box in yolo_results.boxes.data.tolist() 
                               if int(box[5]) == 0 and box[4] >= yolo_conf]
                
                # Process each detected person
                person_found = False
                
                for box in person_boxes:
                    x1, y1, x2, y2, conf, cls = box
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Draw YOLO bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Crop the person from the frame
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                    
                    # Process with MediaPipe
                    frame_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    
                    if results.pose_landmarks:
                        person_found = True
                        consecutive_failures = 0
                        
                        # Store in history for smoothing
                        if len(landmark_history) >= max_history:
                            landmark_history.pop(0)  # Remove oldest
                        landmark_history.append(results.pose_landmarks)
                        
                        last_landmarks = results.pose_landmarks
                        
                        # Draw pose landmarks on the cropped region
                        mp_drawing.draw_landmarks(
                            person_crop,
                            last_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        # Record landmark data (normalized to the crop region)
                        landmarks_list = []
                        for idx, landmark in enumerate(last_landmarks.landmark):
                            landmarks_list.append({
                                "index": idx,
                                "name": mp_pose.PoseLandmark(idx).name,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": landmark.visibility,
                                "crop_x1": x1,  # Store crop coordinates
                                "crop_y1": y1,
                                "crop_x2": x2,
                                "crop_y2": y2
                            })
                        
                        # Apply additional temporal smoothing
                        if len(landmarks_history_raw) >= max_history:
                            landmarks_history_raw.pop(0)
                        landmarks_history_raw.append(landmarks_list)
                        
                        # Apply smoothing if we have enough history
                        if len(landmarks_history_raw) > 1:
                            smoothed_landmarks = []
                            for i, curr_lm in enumerate(landmarks_list):
                                x_sum = curr_lm['x']
                                y_sum = curr_lm['y']
                                z_sum = curr_lm['z']
                                count = 1
                                
                                # Average with historical landmarks
                                for history in landmarks_history_raw[:-1]:
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
                                    "visibility": curr_lm["visibility"],
                                    "crop_x1": x1,
                                    "crop_y1": y1,
                                    "crop_x2": x2,
                                    "crop_y2": y2
                                })
                            
                            landmarks_list = smoothed_landmarks
                        
                        frame_data["landmarks"] = landmarks_list
                        frame_data["bbox"] = [x1, y1, x2, y2]
                        break  # Just use the first person with successful pose detection
                
                if not person_found:
                    consecutive_failures += 1
                    
                    if last_landmarks is not None and consecutive_failures < max_failures:
                        # Mark these as interpolated landmarks in our data
                        landmarks_list = []
                        for idx, landmark in enumerate(last_landmarks.landmark):
                            # Calculate fade factor - gradually reduce visibility for consecutive failures
                            fade_factor = max(0.5, 1.0 - (consecutive_failures * 0.05))
                            
                            landmarks_list.append({
                                "index": idx,
                                "name": mp_pose.PoseLandmark(idx).name,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": landmark.visibility * fade_factor,
                                "interpolated": True
                            })
                        
                        frame_data["landmarks"] = landmarks_list
                        frame_data["interpolated"] = True
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds() * 1000  # in ms
                processing_times.append(processing_time)
                
                # Add processing time to frame_data
                frame_data["processing_time_ms"] = processing_time
                
                # Display processing information on the frame
                cv2.putText(
                    frame, 
                    f"YOLOv8+MP | {processing_time:.1f}ms | Frame {frame_count} | Persons: {len(person_boxes)}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Store landmark data for processed frames
                all_landmarks.append(frame_data)
            
            # Display the frame
            cv2.imshow('Video with YOLOv8 + Pose Detection', frame)
            
            frame_count += 1
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            frame_count += 1
            consecutive_failures += 1
            if consecutive_failures > 10:  # If we can't process even 10 frames in a row, exit
                break
    
    # Save landmarks to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    log_file = os.path.join(log_dir, f"{video_name}_pose_landmarks_{timestamp}.json")
    
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    try:
        with open(log_file, 'w') as f:
            json.dump({
                "video_path": video_path,
                "total_frames": frame_count,
                "processed_frames": len(all_landmarks),
                "processing_rate": f"Every {process_every_n} frames",
                "smoothing_enabled": smooth_landmarks,
                "smoothing_factor": smoothing_factor,
                "yolo_confidence": yolo_conf,
                "detection_confidence": detection_confidence,
                "tracking_confidence": tracking_confidence,
                "avg_processing_time_ms": avg_processing_time,
                "landmarks": all_landmarks
            }, f, indent=2)
        
        print(f"Landmark data saved to: {log_file}")
        print(f"Average processing time: {avg_processing_time:.2f} ms per frame")
    except Exception as e:
        print(f"Error saving landmark data: {e}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read video and detect human pose')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save landmark logs')
    parser.add_argument('--no_smooth', action='store_false', dest='smooth', 
                        help='Disable landmark smoothing')
    parser.add_argument('--process_rate', type=int, default=1, 
                        help='Process every n frames (1=all frames, 2=every other frame, etc.)')
    parser.add_argument('--detection_conf', type=float, default=0.5,
                        help='MediaPipe detection confidence threshold (0.0-1.0)')
    parser.add_argument('--tracking_conf', type=float, default=0.5,
                        help='MediaPipe tracking confidence threshold (0.0-1.0)')
    parser.add_argument('--yolo_conf', type=float, default=0.5,
                        help='YOLO confidence threshold (0.0-1.0)')
    parser.add_argument('--smoothing_factor', type=float, default=0.7,
                        help='Temporal smoothing factor (0.0-1.0)')
    args = parser.parse_args()
    
    # Make sure the path is absolute
    video_path = args.video_path
    if not video_path.startswith("rtsp://") and not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    read_video_with_pose(
        video_path, 
        args.log_dir, 
        args.smooth, 
        args.process_rate,
        args.detection_conf,
        args.tracking_conf,
        args.yolo_conf,
        args.smoothing_factor
    )