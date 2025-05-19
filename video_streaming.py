import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
import signal
import sys
from datetime import datetime

# Global variables for graceful shutdown
running = True
all_landmarks = []
frame_count = 0
processing_times = []

def signal_handler(sig, frame):
    """Handle Ctrl+C to ensure we save data before exiting"""
    global running
    print("\nCtrl+C detected. Saving data before exiting...")
    running = False
    # The main loop will detect this and exit gracefully

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Add resize_factor parameter
def read_stream_with_pose(stream_url, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                       detection_confidence=0.5, tracking_confidence=0.5,
                       log_interval_ms=300, resize_factor=1.0):
    
    # In the frame processing loop:
    if resize_factor != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    global running, all_landmarks, frame_count, processing_times
    
    # Reset globals in case function is called multiple times
    running = True
    all_landmarks = []
    frame_count = 0
    processing_times = []
    
    print(f"Using OpenCV version: {cv2.__version__}")
    # Define time logging
    last_log_time = datetime.now()
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # Use lowest complexity (0) for speed
        enable_segmentation=False,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        smooth_landmarks=smooth_landmarks
    )
    
    # Try opening the video stream
    try:
        print(f"Attempting to connect to stream: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"Error: Failed to open stream {stream_url}.")
            return
    except Exception as e:
        print(f"Error opening stream: {e}")
        return
    
    print(f"Stream connected: {stream_url}")
    print(f"Frame width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"Frame height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Main video processing code
    # Create a list to store landmarks for each frame
    all_landmarks = []
    
    # For smoothing landmarks between frames when there's no detection
    last_landmarks = None
    landmark_history = []
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

    retry_count = 0
    max_retries = 3    
        
    # Main processing loop
    while running:
        try:
            ret, frame = cap.read()
            
            # If we've lost connection, try to reconnect
            if not ret:
                retry_count += 1
                print(f"Lost connection. Retry {retry_count}/{max_retries}")
                if retry_count > max_retries:
                    print("Maximum retries reached. Exiting.")
                    break
                    
                # Try to reconnect
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    print("Failed to reconnect")
                    break
                continue
                
            # Reset retry count on successful frame read
            retry_count = 0
            
            # Process every n frames
            process_this_frame = frame_count % process_every_n == 0
            
            frame_data = {"frame": frame_count, "landmarks": None}
            
            if process_this_frame:
                start_time = datetime.now()
                
                # Convert the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = pose.process(frame_rgb)
                
                # Process landmarks
                if results.pose_landmarks:
                    # Reset failure counter on successful detection
                    consecutive_failures = 0
                    
                    # Store in history for smoothing
                    if len(landmark_history) >= max_history:
                        landmark_history.pop(0)  # Remove oldest
                    landmark_history.append(results.pose_landmarks)
                    
                    # Use the landmarks as-is for now
                    last_landmarks = results.pose_landmarks
                    
                    # Draw pose landmarks - SKIP EYE LANDMARKS
                    mp_drawing.draw_landmarks(
                        frame,
                        last_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Record landmark data - SKIP EYE LANDMARKS
                    landmarks_list = []
                    for idx, landmark in enumerate(last_landmarks.landmark):
                        # Skip eye landmarks (indices 1-6)
                        if 1 <= idx <= 6:
                            continue
                            
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
                    # Handle no detection
                    consecutive_failures += 1
                    
                    if last_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            frame,
                            last_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        # Mark these as interpolated landmarks in our data
                        landmarks_list = []
                        for idx, landmark in enumerate(last_landmarks.landmark):
                            # Skip eye landmarks (indices 1-6)
                            if 1 <= idx <= 6:
                                continue
                                
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
                    f"CPU mode | {processing_time:.1f}ms | Frame {frame_count}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                
                # Store landmark data for processed frames, but only if enough time has passed
                current_time = datetime.now()
                time_since_last_log = (current_time - last_log_time).total_seconds() * 1000

                # Only log if enough time has passed AND we have good detection
                if time_since_last_log >= log_interval_ms:
                    # Check if we have good detection of limbs before logging
                    if has_good_detection(frame_data["landmarks"], min_visibility_threshold=0.6):
                        all_landmarks.append(frame_data)
                        last_log_time = current_time
                        
                        # Add a note that this frame was logged
                        cv2.putText(
                            frame, 
                            "Logged", 
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (0, 255, 255), 
                            2
                        )
                        
                        # Periodically save to disk every 300 frames
                        if frame_count % 300 == 0 and frame_count > 0:
                            save_landmarks_to_file(stream_url, log_dir, frame_count, all_landmarks, 
                                                process_every_n, smooth_landmarks, detection_confidence, 
                                                tracking_confidence, processing_times, is_final=False)
                    else:
                        # Mark frames with poor detections (for debugging)
                        cv2.putText(
                            frame, 
                            "Poor detection - not logged", 
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (0, 0, 255),  # Red color
                            2
                        )
                
                # Store landmark data for processed frames
                # all_landmarks.append(frame_data)
            
            # Display the frame
            cv2.imshow('Video with Pose Detection', frame)
            
            frame_count += 1
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            frame_count += 1
            if frame_count > 10:  # If we can't process even the first 10 frames, exit
                break
    
    # Save landmarks to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rtsp_stream_pose_landmarks_NEW_{timestamp}.json")

    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

    # Always save data, regardless of how we exited the loop
    try:
        with open(log_file, 'w') as f:
            json.dump({
                "stream_url": stream_url,
                "total_frames": frame_count,
                "processed_frames": len(all_landmarks),
                "processing_rate": f"Every {process_every_n} frames",
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
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

def save_landmarks_to_file(stream_url, log_dir, frame_count, landmarks, process_rate, 
                          smooth_enabled, detection_conf, tracking_conf, 
                          proc_times, is_final=True):
    """Save landmarks to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "final" if is_final else "partial"
    log_file = os.path.join(log_dir, f"rtsp_stream_pose_landmarks_{suffix}_{timestamp}.json")

    avg_time = sum(proc_times) / len(proc_times) if proc_times else 0
    
    try:
        with open(log_file, 'w') as f:
            json.dump({
                "stream_url": stream_url,
                "total_frames": frame_count,
                "processed_frames": len(landmarks),
                "processing_rate": f"Every {process_rate} frames",
                "smoothing_enabled": smooth_enabled,
                "detection_confidence": detection_conf,
                "tracking_confidence": tracking_conf,
                "avg_processing_time_ms": avg_time,
                "landmarks": landmarks
            }, f, indent=2)
        
        print(f"Landmark data saved to: {log_file}")
        print(f"Average processing time: {avg_time:.2f} ms per frame")
    except Exception as e:
        print(f"Error saving landmark data: {e}")

# Add this function to check if key body parts are detected

def has_good_detection(landmarks_data, min_visibility_threshold=0.5):
    """
    Check if the frame has good detection of key body parts
    
    Args:
        landmarks_data: List of landmark dictionaries
        min_visibility_threshold: Minimum visibility score required
        
    Returns:
        bool: True if all key body parts have good visibility
    """
    if not landmarks_data:
        return False
    
    # Define key body parts to check
    key_landmarks = {
        # Hands
        "LEFT_WRIST": False,
        "RIGHT_WRIST": False,
        # Legs
        "LEFT_KNEE": False,
        "RIGHT_KNEE": False,
        "LEFT_ANKLE": False,
        "RIGHT_ANKLE": False,
    }
    
    # Check each landmark
    for landmark in landmarks_data:
        name = landmark.get("name", "")
        visibility = landmark.get("visibility", 0)
        index = landmark.get("index", -1)
        
        # Skip eye landmarks (indices 1-6)
        if 1 <= index <= 6:
            continue
            
        if name in key_landmarks and visibility >= min_visibility_threshold:
            key_landmarks[name] = True
    
    # We need at least one good hand and one good leg
    has_hand = key_landmarks["LEFT_WRIST"] or key_landmarks["RIGHT_WRIST"]
    has_leg = (key_landmarks["LEFT_KNEE"] and key_landmarks["LEFT_ANKLE"]) or \
              (key_landmarks["RIGHT_KNEE"] and key_landmarks["RIGHT_ANKLE"])
    
    return has_hand and has_leg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read RTSP stream and detect human pose')
    parser.add_argument('stream_url', type=str, nargs='?',
                        default="rtsp://admin:Khoa12345^@192.168.1.17/cam/realmonitor?channel=1&subtype=0",
                        help='RTSP stream URL')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save landmark logs')
    parser.add_argument('--no_smooth', action='store_false', dest='smooth', 
                        help='Disable landmark smoothing')
    parser.add_argument('--process_rate', type=int, default=1, 
                        help='Process every n frames (1=all frames, 2=every other frame, etc.)')
    parser.add_argument('--detection_conf', type=float, default=0.5,
                        help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--tracking_conf', type=float, default=0.5,
                        help='Tracking confidence threshold (0.0-1.0)')
    parser.add_argument('--log_interval', type=int, default=300,
                        help='Minimum interval between logged frames (milliseconds)')
    args = parser.parse_args()
    
    read_stream_with_pose(
        args.stream_url, 
        args.log_dir, 
        args.smooth, 
        args.process_rate,
        args.detection_conf,
        args.tracking_conf,
        args.log_interval
    )