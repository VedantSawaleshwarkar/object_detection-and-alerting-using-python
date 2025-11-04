import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
from datetime import datetime
from pathlib import Path

print("1. Imported all required libraries")

# Initialize video stream
print("2. Initializing video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Warm up camera
print("3. Video stream started")

# Load face detector
print("4. Loading face detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("5. Loaded face detector")

# Load face recognizer
print("6. Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
print("7. Loaded face recognizer")

# Load the actual face recognition model
print("8. Loading recognition model...")
try:
    recognizer = pickle.loads(open("output/recognizer", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())
    print("9. Loaded recognition model and label encoder")
except Exception as e:
    print(f"Error loading recognition model: {e}")
    exit(1)

print("10. Starting main loop...")

# Start the FPS counter
fps = FPS().start()

# Main loop
while True:
    try:
        # Grab the frame from the video stream
        frame = vs.read()
        if frame is None:
            print("Error: Could not read frame from camera")
            break
            
        # Resize the frame
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        
        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Break the loop if 'q' is pressed
        if key == ord("q"):
            break
            
        # Update the FPS counter
        fps.update()
        
    except Exception as e:
        print(f"Error in main loop: {e}")
        break

# Stop the timer and display FPS information
fps.stop()
print(f"Elapsed time: {fps.elapsed():.2f}")
print(f"Approx. FPS: {fps.fps():.2f}")

# Clean up
cv2.destroyAllWindows()
vs.stop()
print("Done")
