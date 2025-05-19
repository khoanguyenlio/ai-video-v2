import os
import requests
import zipfile
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

def download_pose_landmarker_model():
    """
    Download the pose landmarker model from the MediaPipe repository
    and save it to the local directory
    """
    print(f"Downloading pose landmarker model...")
    
    try:
        # Create directories if they don't exist
        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Download the model file
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        # Save the model file
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded successfully to {MODEL_PATH}")
        print(f"Ready to use with video_reader_webcam.py")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download the model manually from:")
        print("https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index")
        print(f"and save it as {MODEL_PATH}")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Model file already exists at {MODEL_PATH}")
        print("Ready to use with video_reader_webcam.py")
    else:
        download_pose_landmarker_model()