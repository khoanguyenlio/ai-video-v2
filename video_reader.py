import cv2
import numpy as np
import argparse
import mediapipe as mp
import json
import os
from datetime import datetime

def read_video_with_pose(video_path, log_dir="logs", smooth_landmarks=True, process_every_n=1, 
                         detection_confidence=0.5, tracking_confidence=0.5):
    """
    Read a video file and detect human pose using MediaPipe
    
    Args:
        video_path (str): Path to the video file
        log_dir (str): Directory to save landmark logs
        smooth_landmarks (bool): Whether to smooth landmarks across frames
        process_every_n (int): Process every nth frame (lower values = more processing but smoother results)
        detection_confidence (float): Minimum detection confidence threshold
        tracking_confidence (float): Minimum tracking confidence threshold
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
            # Apply preprocessing to enhance the frame if needed
            if consecutive_failures > 3:
                # Apply preprocessing to enhance detection of left side
                # Inside read_video_with_pose function, modify the part where you process frames:

                # Always apply left-side enhancement for more consistent detection
                enhanced_frame = enhance_frame_for_left_side(frame)
                frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

                # After detecting landmarks, add this before drawing:
                if results.pose_landmarks:
                    # Always try to improve the left side even on successful detection
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
    
    with open(log_file, 'w') as f:
        json.dump({
            "video_path": video_path,
            "total_frames": frame_count,
            "processed_frames": len(all_landmarks),
            "processing_rate": f"Every {process_every_n} frames",
            "smoothing_enabled": smooth_landmarks,
            "detection_confidence": detection_confidence,
            "tracking_confidence": tracking_confidence,
            "landmarks": all_landmarks
        }, f, indent=2)
    
    print(f"Landmark data saved to: {log_file}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

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
        31: [27, 29],  # LEFT_FOOT_INDEX depends on LEFT_ANKLE and LEFT_HEEL
        17: [15, 13],  # LEFT_PINKY depends on LEFT_WRIST and LEFT_ELBOW
        19: [15, 13],  # LEFT_INDEX depends on LEFT_WRIST and LEFT_ELBOW
        21: [15, 13]   # LEFT_THUMB depends on LEFT_WRIST and LEFT_ELBOW
    }
    
    # For each problem landmark
    for problem_idx, related_indices in problem_indices.items():
        current = recovered.landmark[problem_idx]
        
        # If current visibility is poor, try to recover
        if current.visibility < 0.7:  # Increased threshold to catch more low-confidence detections
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
                recovered.landmark[problem_idx].visibility = best_visibility * 0.9
            else:
                # Try to infer from related landmarks if they have good visibility
                related_landmarks = [recovered.landmark[idx] for idx in related_indices]
                
                # Relaxed the visibility requirement for related landmarks
                if any(lm.visibility > 0.6 for lm in related_landmarks):
                    # LEFT_WRIST from LEFT_ELBOW
                    if problem_idx == 15 and related_landmarks[0].visibility > 0.6:
                        elbow = related_landmarks[0]
                        shoulder = related_landmarks[1]
                        
                        # Calculate the direction from shoulder to elbow
                        dx = elbow.x - shoulder.x
                        dy = elbow.y - shoulder.y
                        
                        # Extend from elbow in that direction
                        recovered.landmark[problem_idx].x = elbow.x + dx * 0.9
                        recovered.landmark[problem_idx].y = elbow.y + dy * 0.9
                        recovered.landmark[problem_idx].visibility = 0.65
                    
                    # LEFT_KNEE from hips
                    elif problem_idx == 25 and all(lm.visibility > 0.6 for lm in related_landmarks):
                        left_hip = related_landmarks[0]
                        right_hip = related_landmarks[1]
                        
                        # Use anatomical knowledge to estimate knee position
                        # Assume knee is below and slightly out from hip
                        recovered.landmark[problem_idx].x = left_hip.x + 0.05
                        recovered.landmark[problem_idx].y = left_hip.y + 0.2
                        recovered.landmark[problem_idx].visibility = 0.65
                    
                    # LEFT_ANKLE from LEFT_KNEE and LEFT_HIP
                    elif problem_idx == 27 and related_landmarks[0].visibility > 0.6:
                        knee = related_landmarks[0]
                        hip = related_landmarks[1]
                        
                        # Calculate direction from hip to knee
                        dx = knee.x - hip.x
                        dy = knee.y - hip.y
                        
                        # Extend from knee in that direction
                        recovered.landmark[problem_idx].x = knee.x + dx * 0.9
                        recovered.landmark[problem_idx].y = knee.y + dy * 0.9
                        recovered.landmark[problem_idx].visibility = 0.65
                    
                    # Finger landmarks from wrist position
                    elif problem_idx in [17, 19, 21] and related_landmarks[0].visibility > 0.6:
                        wrist = related_landmarks[0]
                        elbow = related_landmarks[1]
                        
                        # Direction from elbow to wrist
                        dx = wrist.x - elbow.x
                        dy = wrist.y - elbow.y
                        
                        # Different offsets for different fingers
                        if problem_idx == 17:  # LEFT_PINKY
                            offset_x = -0.01  # Slightly to the left
                            offset_y = 0.03   # Slightly down
                        elif problem_idx == 19:  # LEFT_INDEX
                            offset_x = 0.01   # Slightly to the right
                            offset_y = 0.03   # Slightly down
                        else:  # LEFT_THUMB
                            offset_x = 0.02   # More to the right
                            offset_y = 0.01   # Less down
                        
                        recovered.landmark[problem_idx].x = wrist.x + dx * 0.4 + offset_x
                        recovered.landmark[problem_idx].y = wrist.y + dy * 0.4 + offset_y
                        recovered.landmark[problem_idx].visibility = 0.65
    
    return recovered

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
    
    read_video_with_pose(
        args.video_path, 
        args.log_dir, 
        args.smooth, 
        args.process_rate,
        args.detection_conf,
        args.tracking_conf
    )