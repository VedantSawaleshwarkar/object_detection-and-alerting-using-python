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

print("Starting face recognition...")

# Load serialized face detector
print("Loading face detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the actual face recognition model along with the label encoder
print("Loading recognition model...")
try:
    recognizer = pickle.loads(open("output/recognizer", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())
except Exception as e:
    print(f"Error loading recognition model: {e}")
    exit(1)

# Initialize the video stream, then allow the camera sensor to warm up
print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

print("Press 'q' to quit")

# Loop over frames from the video file stream
while True:
    try:
        # Grab the frame from the threaded video stream
        frame = vs.read()
        if frame is None:
            print("Error: Could not read frame from camera")
            break

        # Resize the frame to have a width of 600 pixels
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # Construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Apply OpenCV's deep learning-based face detector
        detector.setInput(imageBlob)
        detections = detector.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:
                # Compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # Extract the face ROI and draw a rectangle around the face
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # Draw a rectangle around the face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = "Face: {:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Show the output frame
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
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
