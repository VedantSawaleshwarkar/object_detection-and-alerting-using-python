import argparse
import cv2
from ultralytics import YOLO

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Load the YOLOv8 model pretrained on COCO
model = YOLO("yolov8n.pt")

# Load the image
image = cv2.imread(args["image"])

# Perform inference
results = model(image)

# Display the results
results[0].show()