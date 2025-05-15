import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
from datetime import datetime
import sys

def read_video_with_pose(video_path, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                         detection_confidence=0.5, tracking_confidence=0.5):
    """
    Read a video file and detect human pose using MediaPipe
    """
    # First verify the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    print(f"Using OpenCV version: {cv2.__version__}")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
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
                
                # Store landmark data for processed frames
                all_landmarks.append(frame_data)
            
            # Display the frame
            cv2.imshow('Video with Pose Detection', frame)
            
            frame_count += 1
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            frame_count += 1
            if frame_count > 10:  # If we can't process even the first 10 frames, exit
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
                        help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--tracking_conf', type=float, default=0.5,
                        help='Tracking confidence threshold (0.0-1.0)')
    args = parser.parse_args()
    
    # Make sure the path is absolute
    video_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    read_video_with_pose(
        video_path, 
        args.log_dir, 
        args.smooth, 
        args.process_rate,
        args.detection_conf,
        args.tracking_conf
    )