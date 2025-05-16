import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
from datetime import datetime
import time

def read_stream_with_pose(stream_source=0, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                          detection_confidence=0.5, tracking_confidence=0.5, max_time=None):
    """
    Read from a video stream and detect human pose using MediaPipe
    
    Args:
        stream_source: Camera index or stream URL (0 is usually the default camera)
        log_dir (str): Directory to save landmark logs
        smooth_landmarks (bool): Whether to smooth landmarks across frames
        process_every_n (int): Process every nth frame (lower values = more processing but smoother results)
        detection_confidence (float): Minimum detection confidence threshold
        tracking_confidence (float): Minimum tracking confidence threshold
        max_time (float): Maximum time to capture in seconds (None = run until manually stopped)
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Using higher complexity for better detection
        enable_segmentation=False,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        smooth_landmarks=smooth_landmarks  # Enable MediaPipe's built-in smoothing
    )
    
    # Open the video stream
    cap = cv2.VideoCapture(stream_source)
    
    # Check if stream opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open stream source {stream_source}")
        return
    
    print(f"Stream opened: {stream_source}")
    print(f"Frame width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"Frame height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Create a list to store landmarks for each frame
    all_landmarks = []
    
    # For smoothing landmarks between frames when there's no detection
    last_landmarks = None
    landmark_history = []  # Store multiple frames of landmarks for better smoothing
    max_history = 5  # Max frames to keep in history
    
    # For stabilizing detection
    consecutive_failures = 0
    max_failures = 10  # Max consecutive failures before more aggressive recovery
    
    # For tracking timing
    start_time = time.time()
    
    # Read and display video frames
    frame_count = 0
    while cap.isOpened():
        # Check if we've reached the maximum time
        if max_time is not None and (time.time() - start_time) > max_time:
            print(f"Reached maximum capture time of {max_time} seconds")
            break
            
        ret, frame = cap.read()
        
        # Break the loop if there's an issue with the stream
        if not ret:
            print("Stream ended or error occurred")
            break
        
        # Process every n frames to balance performance and smoothness
        process_this_frame = frame_count % process_every_n == 0
        
        frame_data = {"frame": frame_count, "landmarks": None, "timestamp": time.time()}
        
        # Always convert the frame for display
        if process_this_frame:
            # Apply preprocessing to enhance the frame if needed
            if consecutive_failures > 3:
                # Apply preprocessing to enhance detection of left side
                enhanced_frame = enhance_frame_for_left_side(frame)
                frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

                # Process the enhanced frame
                results = pose.process(frame_rgb)
                
                # After detecting landmarks, improve the left side
                if results.pose_landmarks:
                    try:
                        improved_landmarks = attempt_left_side_recovery(results.pose_landmarks, landmark_history)
                        results.pose_landmarks = improved_landmarks
                    except Exception as e:
                        print(f"Error improving landmarks: {e}")
            else:
                # Convert the BGR image to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe Pose
                results = pose.process(frame_rgb)
            
            # If landmarks were detected, update last_landmarks
            if results.pose_landmarks:
                # Reset failure counter on successful detection
                consecutive_failures = 0
                
                # Store in history for smoothing
                if len(landmark_history) >= max_history:
                    landmark_history.pop(0)  # Remove oldest
                landmark_history.append(results.pose_landmarks)
                
                # Apply temporal smoothing if we have history
                if len(landmark_history) > 1 and smooth_landmarks:
                    smoothed_landmarks = apply_temporal_smoothing(
                        results.pose_landmarks, 
                        landmark_history
                    )
                    last_landmarks = smoothed_landmarks
                else:
                    last_landmarks = results.pose_landmarks
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    last_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Record landmark data
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
                
                frame_data["landmarks"] = landmarks_list
            else:
                # Increment failure counter on failed detection
                consecutive_failures += 1
                
                # If no detection in this frame but we have previous landmarks, use those for visualization
                if last_landmarks is not None:
                    # For long sequences of failures, try to enhance left-side detection
                    if consecutive_failures > max_failures and len(landmark_history) > 0:
                        # Apply more aggressive recovery techniques
                        try:
                            # Attempt to recover left side landmarks from history
                            recovered_landmarks = attempt_left_side_recovery(last_landmarks, landmark_history)
                            if recovered_landmarks:
                                last_landmarks = recovered_landmarks
                        except Exception as e:
                            print(f"Error recovering landmarks: {e}")
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        last_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
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
            
            # Only store landmark data for processed frames
            all_landmarks.append(frame_data)
        
        # Display the frame regardless of whether it was processed
        cv2.imshow('Stream with Pose Detection', frame)
        
        frame_count += 1
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save landmarks to a JSON file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"stream_pose_landmarks_{timestamp}.json")
    
    with open(log_file, 'w') as f:
        json.dump({
            "stream_source": stream_source,
            "total_frames": frame_count,
            "processed_frames": len(all_landmarks),
            "processing_rate": f"Every {process_every_n} frames",
            "smoothing_enabled": smooth_landmarks,
            "detection_confidence": detection_confidence,
            "tracking_confidence": tracking_confidence,
            "duration": time.time() - start_time,
            "landmarks": all_landmarks
        }, f, indent=2)
    
    print(f"Landmark data saved to: {log_file}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# Import the functions from video_reader.py to avoid duplication
from video_reader import enhance_frame_for_left_side, attempt_left_side_recovery, apply_temporal_smoothing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read from a video stream and detect human pose')
    parser.add_argument('--source', type=int, default=0, 
                        help='Camera index or stream URL (0 is usually the default camera)')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Directory to save landmark logs')
    parser.add_argument('--no_smooth', action='store_false', dest='smooth', 
                        help='Disable landmark smoothing')
    parser.add_argument('--process_rate', type=int, default=1, 
                        help='Process every n frames (1=all frames, 2=every other frame, etc.)')
    parser.add_argument('--detection_conf', type=float, default=0.5,
                        help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--tracking_conf', type=float, default=0.5,
                        help='Tracking confidence threshold (0.0-1.0)')
    parser.add_argument('--max_time', type=float, default=None,
                        help='Maximum time to capture in seconds')
    args = parser.parse_args()
    
    read_stream_with_pose(
        args.source, 
        args.log_dir, 
        args.smooth, 
        args.process_rate,
        args.detection_conf,
        args.tracking_conf,
        args.max_time
    )