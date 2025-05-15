import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
os.environ['MEDIAPIPE_USE_GPU'] = '1'
from datetime import datetime
import tensorflow as tf
import sys

def read_video_with_pose(video_path, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                         detection_confidence=0.5, tracking_confidence=0.5, use_gpu=True):
    """
    Read a video file and detect human pose using MediaPipe with GPU acceleration when possible
    """
    # First verify the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    # Check if CUDA is available
    cuda_available = False
    if use_gpu:
        try:
            cuda_available = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
            if cuda_available:
                print("GPU acceleration enabled for OpenCV operations")
            else:
                print("GPU acceleration requested but CUDA not available in OpenCV. Using CPU.")
        except Exception as e:
            print(f"Error checking CUDA availability: {e}")
            print("Using CPU for OpenCV operations")
    else:
        print("Using CPU for OpenCV operations")
    
    # Check if TensorFlow can see GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"MediaPipe will use GPU: {physical_devices}")
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error configuring TensorFlow GPU: {e}")
    else:
        print("No GPU detected for MediaPipe/TensorFlow")
    
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
    
    # Continue with the rest of your code...
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
    while cap.isOpened():
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
                
                # Skip CUDA-specific code since it's not available - just use CPU processing
                # Convert the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = pose.process(frame_rgb)
                
                # Rest of your code for handling pose landmarks...
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
                    # Handle landmark interpolation logic...
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
                "gpu_acceleration": False,  # We're using CPU mode
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

# Keep the original helper functions
def enhance_frame_for_left_side(frame):
    # ... [keep your existing function]
    """Enhance frame to improve detection of the left side body parts"""
    # Create a copy of the frame
    enhanced = frame.copy()
    
    # Split into RGB channels
    b, g, r = cv2.split(enhanced)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    
    # Merge the CLAHE enhanced channels
    enhanced = cv2.merge([b, g, r])
    
    # Divide the image into left and right halves
    height, width = enhanced.shape[:2]
    midpoint = width // 2
    
    # We want a bit more than just the left half for context
    left_section = enhanced[:, :int(midpoint*1.2)]
    
    # Apply stronger enhancement to the left section
    alpha = 1.3  # Increase contrast more (1.0 means no change)
    beta = 15    # Increase brightness more (0 means no change)
    left_section = cv2.convertScaleAbs(left_section, alpha=alpha, beta=beta)
    
    # Put the enhanced left section back
    enhanced[:, :int(midpoint*1.2)] = left_section
    
    # Apply slight blur to reduce noise but preserve edges
    enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
    
    return enhanced

def apply_temporal_smoothing(current_landmarks, landmark_history):
    # ... [keep your existing function]
    """Apply temporal smoothing across multiple frames of landmarks"""
    import copy
    smoothed = copy.deepcopy(current_landmarks)
    
    # LEFT_WRIST = 15, LEFT_ELBOW = 13, LEFT_KNEE = 25, LEFT_ANKLE = 27, LEFT_FOOT_INDEX = 31
    left_side_indices = [13, 15, 17, 19, 21, 25, 27, 29, 31]
    
    # For each landmark
    for i in range(len(smoothed.landmark)):
        # Get current landmark
        curr = current_landmarks.landmark[i]
        
        # Special handling for left-side landmarks
        if i in left_side_indices:
            # Find valid left-side landmark from history (higher weight on recent frames)
            valid_history = []
            weights = []
            
            for idx, hist_landmarks in enumerate(reversed(landmark_history)):
                hist_landmark = hist_landmarks.landmark[i]
                if hist_landmark.visibility > 0.3:  # Only use reasonably visible historical landmarks
                    valid_history.append(hist_landmark)
                    # Higher weight for more recent frames (0.8, 0.6, 0.4, 0.2 for last 4 frames)
                    weights.append(max(0.2, 0.8 - (idx * 0.2)))
            
            # If we have valid history, apply weighted average
            if valid_history:
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    norm_weights = [w/total_weight for w in weights]
                    
                    # Apply weighted average
                    smoothed.landmark[i].x = sum(h.x * w for h, w in zip(valid_history, norm_weights))
                    smoothed.landmark[i].y = sum(h.y * w for h, w in zip(valid_history, norm_weights))
                    smoothed.landmark[i].z = sum(h.z * w for h, w in zip(valid_history, norm_weights))
                    
                    # Use max visibility with a slight discount
                    smoothed.landmark[i].visibility = max(h.visibility for h in valid_history) * 0.95
        else:
            # For non-left-side landmarks, use simple smoothing with the previous frame
            if len(landmark_history) > 1:
                prev = landmark_history[-2].landmark[i]  # Previous frame
                alpha = 0.7  # Current frame weight
                
                # If current landmark has low visibility, rely more on previous
                if curr.visibility < 0.5:
                    alpha = 0.3
                
                # Apply weighted average
                smoothed.landmark[i].x = curr.x * alpha + prev.x * (1 - alpha)
                smoothed.landmark[i].y = curr.y * alpha + prev.y * (1 - alpha)
                smoothed.landmark[i].z = curr.z * alpha + prev.z * (1 - alpha)
    
    return smoothed

def attempt_left_side_recovery(current_landmarks, landmark_history):
    # ... [keep your existing function]
    """Attempt to recover left side landmarks using historical data"""
    import copy
    recovered = copy.deepcopy(current_landmarks)
    
    # Define left side landmarks that commonly have issues
    problem_indices = {
        15: [13, 11],  # LEFT_WRIST depends on LEFT_ELBOW and LEFT_SHOULDER
        13: [11, 23],  # LEFT_ELBOW depends on LEFT_SHOULDER and LEFT_HIP
        25: [23, 24],  # LEFT_KNEE depends on LEFT_HIP and RIGHT_HIP
        27: [25, 23],  # LEFT_ANKLE depends on LEFT_KNEE and LEFT_HIP
        31: [27, 29]   # LEFT_FOOT_INDEX depends on LEFT_ANKLE and LEFT_HEEL
    }
    
    # For each problem landmark
    for problem_idx, related_indices in problem_indices.items():
        current = recovered.landmark[problem_idx]
        
        # If current visibility is poor, try to recover
        if current.visibility < 0.6:
            # Check history for better visibility
            best_visibility = current.visibility
            best_landmark = None
            
            # Look through history to find better information
            for hist_landmarks in reversed(landmark_history):
                hist_landmark = hist_landmarks.landmark[problem_idx]
                if hist_landmark.visibility > best_visibility:
                    best_visibility = hist_landmark.visibility
                    best_landmark = hist_landmark
            
            # If we found a better historical position, use it
            if best_landmark and best_visibility > 0.6:
                recovered.landmark[problem_idx].x = best_landmark.x
                recovered.landmark[problem_idx].y = best_landmark.y
                recovered.landmark[problem_idx].z = best_landmark.z
                recovered.landmark[problem_idx].visibility = best_visibility * 0.9  # Slightly reduce
            else:
                # Try to infer from related landmarks if they have good visibility
                related_landmarks = [recovered.landmark[idx] for idx in related_indices]
                if all(lm.visibility > 0.7 for lm in related_landmarks):
                    # Simple heuristic: for wrist and elbow, extend from shoulder
                    if problem_idx == 15 and related_landmarks[0].visibility > 0.7:  # LEFT_WRIST from LEFT_ELBOW
                        elbow = related_landmarks[0]
                        shoulder = related_landmarks[1]
                        
                        # Extend from elbow in the direction from shoulder to elbow
                        dx = elbow.x - shoulder.x
                        dy = elbow.y - shoulder.y
                        
                        recovered.landmark[problem_idx].x = elbow.x + dx * 0.8
                        recovered.landmark[problem_idx].y = elbow.y + dy * 0.8
                        recovered.landmark[problem_idx].visibility = 0.6  # Mark as estimated
    
    return recovered

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
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU acceleration')
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
        args.tracking_conf,
        not args.no_gpu
    )