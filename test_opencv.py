import cv2
import sys

print(f"OpenCV version: {cv2.__version__}")
print(f"Python version: {sys.version}")

# Try to access the VideoCapture class
try:
    cap = cv2.VideoCapture(0)  # Try with camera
    print("VideoCapture is available")
    cap.release()
except Exception as e:
    print(f"Error with VideoCapture: {e}")

# Try to read an image
try:
    img = cv2.imread("demo/unnamed.png")  # Replace with any image path
    if img is not None:
        print("Image reading works")
    else:
        print("Could not read image (returned None)")
except Exception as e:
    print(f"Error reading image: {e}")