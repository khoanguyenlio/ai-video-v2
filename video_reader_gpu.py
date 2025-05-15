import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
os.environ['MEDIAPIPE_USE_GPU'] = '1'
from datetime import datetime
import tensorflow as tf

def read_video_with_pose(video_path, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                         detection_confidence=0.5, tracking_confidence=0.5, use_gpu=True):
    """
    Read a video file and detect human pose using MediaPipe with GPU acceleration when possible
    
    Args:
        video_path (str): Path to the video file
        log_dir (str): Directory to save landmark logs
        smooth_landmarks (bool): Whether to smooth landmarks across frames
        process_every_n (int): Process every nth frame (lower values = more processing but smoother results)
        detection_confidence (float): Minimum detection confidence threshold
        tracking_confidence (float): Minimum tracking confidence threshold
        use_gpu (bool): Whether to use GPU acceleration for OpenCV operations
    """
    # Check if CUDA is available for OpenCV operations
    # First check if cv2.cuda is available
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
    
    # Check if TensorFlow can see GPU (for MediaPipe)
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"MediaPipe will use GPU: {physical_devices}")
        # Configure TensorFlow to use memory growth to avoid allocating all memory
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error configuring TensorFlow GPU: {e}")
    else:
        print("No GPU detected for MediaPipe/TensorFlow")
    
    # Initialize MediaPipe Pose (MediaPipe uses GPU automatically if available)
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
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    print(f"Video opened: {video_path}")
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
    
    # For measuring performance
    processing_times = []
    
    # Read and display video frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Process every n frames to balance performance and smoothness
        process_this_frame = frame_count % process_every_n == 0
        
        frame_data = {"frame": frame_count, "landmarks": None}
        
        # Always convert the frame for display
        if process_this_frame:
            start_time = datetime.now()
            
            # Apply preprocessing to enhance the frame if needed
            if consecutive_failures > 3:
                # Use GPU acceleration for frame enhancement if available
                if use_gpu and cuda_available:
                    try:
                        # Upload frame to GPU
                        gpu_frame = cv2.cuda_GpuMat()
                        gpu_frame.upload(frame)
                        
                        # Apply CUDA-accelerated preprocessing (approximation of enhance_frame_for_left_side)
                        gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
                        channels = cv2.cuda.split(gpu_frame)
                        
                        # Enhance brightness/saturation for the left half
                        h, w = frame.shape[:2]
                        left_mask = np.zeros((h, w), dtype=np.uint8)
                        left_mask[:, :int(w*0.6)] = 255
                        left_mask_gpu = cv2.cuda_GpuMat()
                        left_mask_gpu.upload(left_mask)
                        
                        # Adjust saturation and value
                        saturation_factor = 30
                        value_factor = 20
                        
                        # Apply to saturation channel
                        sat_adjusted = cv2.cuda.add(channels[1], saturation_factor, mask=left_mask_gpu)
                        channels[1] = sat_adjusted
                        
                        # Apply to value channel
                        val_adjusted = cv2.cuda.add(channels[2], value_factor, mask=left_mask_gpu)
                        channels[2] = val_adjusted
                        
                        gpu_frame = cv2.cuda.merge(channels)
                        gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_HSV2BGR)
                        
                        # Apply bilateral filter to reduce noise while preserving edges
                        gpu_frame = cv2.cuda.bilateralFilter(gpu_frame, 5, 75, 75)
                        
                        # Download from GPU
                        enhanced_frame = gpu_frame.download()
                    except Exception as e:
                        print(f"Error using CUDA: {e}")
                        # Fall back to CPU
                        enhanced_frame = enhance_frame_for_left_side(frame)
                else:
                    # CPU fallback
                    enhanced_frame = enhance_frame_for_left_side(frame)
                
                frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            else:
                # Convert the BGR image to RGB for MediaPipe
                if use_gpu and cuda_available:
                    try:
                        # Use GPU for color conversion
                        gpu_frame = cv2.cuda_GpuMat()
                        gpu_frame.upload(frame)
                        gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                        frame_rgb = gpu_frame.download()
                    except Exception as e:
                        print(f"Error using CUDA: {e}")
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
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
                        # Apply more aggressive augmentation techniques
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
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000  # in milliseconds
            processing_times.append(processing_time)
            
            # Add processing time to frame_data
            frame_data["processing_time_ms"] = processing_time
            
            # Display processing information on the frame
            cv2.putText(
                frame, 
                f"GPU: {'Yes' if (use_gpu and cuda_available) else 'No'} | {processing_time:.1f}ms", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
            
            # Only store landmark data for processed frames
            all_landmarks.append(frame_data)
        
        # Display the frame regardless of whether it was processed
        cv2.imshow('Video with Pose Detection', frame)
        
        frame_count += 1
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save landmarks to a JSON file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    log_file = os.path.join(log_dir, f"{video_name}_pose_landmarks_{timestamp}.json")
    
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    with open(log_file, 'w') as f:
        json.dump({
            "video_path": video_path,
            "total_frames": frame_count,
            "processed_frames": len(all_landmarks),
            "processing_rate": f"Every {process_every_n} frames",
            "smoothing_enabled": smooth_landmarks,
            "detection_confidence": detection_confidence,
            "tracking_confidence": tracking_confidence,
            "gpu_acceleration": use_gpu and cuda_available,
            "avg_processing_time_ms": avg_processing_time,
            "landmarks": all_landmarks
        }, f, indent=2)
    
    print(f"Landmark data saved to: {log_file}")
    print(f"Average processing time: {avg_processing_time:.2f} ms per frame")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# Keep the original functions unchanged
def enhance_frame_for_left_side(frame):
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
    parser = argparse.ArgumentParser(description='Read video and detect human pose with GPU acceleration')
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
    
    read_video_with_pose(
        args.video_path, 
        args.log_dir, 
        args.smooth, 
        args.process_rate,
        args.detection_conf,
        args.tracking_conf,
        not args.no_gpu
    )