# import libraries
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

# project config and notifier
try:
    from config import UNKNOWN_THRESHOLD, ALERT_COOLDOWN_SECONDS, UNKNOWN_CLIP_SECONDS
except Exception:
    UNKNOWN_THRESHOLD = 0.7
    ALERT_COOLDOWN_SECONDS = 15
    UNKNOWN_CLIP_SECONDS = 5

try:
    from config import UNKNOWN_CLIPS_DIR
except Exception:
    UNKNOWN_CLIPS_DIR = "output/unknown_clips"

from telegram_notifier import send_text_message, send_video

# additional margin for unknown classification
try:
    from config import MARGIN_MIN
except Exception:
    MARGIN_MIN = 0.15

# absolute probability gate
try:
    from config import ABS_MIN_PROB
except Exception:
    ABS_MIN_PROB = 0.50

def ensure_dir(path_str):
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p

# Optional face enhancement settings
try:
    from config import (
        ENHANCE_FACE,
        CLAHE_CLIP_LIMIT,
        CLAHE_TILE_GRID_SIZE,
        UNSHARP_AMOUNT,
        UNSHARP_KERNEL,
        UNSHARP_SIGMA,
        BBOX_EXPAND_RATIO,
    )
except Exception:
    ENHANCE_FACE = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    UNSHARP_AMOUNT = 1.0
    UNSHARP_KERNEL = (0, 0)
    UNSHARP_SIGMA = 1.0
    BBOX_EXPAND_RATIO = 0.10

def enhance_face_roi(roi):
    if not ENHANCE_FACE:
        return roi
    try:
        # Apply CLAHE on Y channel in YCrCb
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        y = clahe.apply(y)
        ycrcb = cv2.merge((y, cr, cb))
        clahe_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        # Unsharp masking
        if UNSHARP_AMOUNT > 0:
            blurred = cv2.GaussianBlur(clahe_bgr, UNSHARP_KERNEL, UNSHARP_SIGMA)
            sharp = cv2.addWeighted(clahe_bgr, 1 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0)
            return sharp
        return clahe_bgr
    except Exception:
        return roi

def record_clip(vs, first_frame, duration_s, save_dir):
    ensure_dir(save_dir)
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_est = 20.0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(save_dir) / f"unknown_{ts}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps_est, (w, h))
    try:
        end_time = time.time() + duration_s
        writer.write(first_frame)
        while time.time() < end_time:
            f = vs.read()
            if f is None:
                break
            f = imutils.resize(f, width=w)
            fh, fw = f.shape[:2]
            if (fw, fh) != (w, h):
                f = cv2.resize(f, (w, h))
            writer.write(f)
            time.sleep(0.03)
    finally:
        writer.release()
    return out_path
try:
    from config import UNKNOWN_THRESHOLD
except Exception:
    UNKNOWN_THRESHOLD = 0.7
try:
    from config import UNKNOWN_THRESHOLD
except Exception:
    UNKNOWN_THRESHOLD = 0.7

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder (with fallback)
try:
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
except Exception as e1:
    try:
        recognizer = pickle.loads(open("output/recognizer", "rb").read())
    except Exception as e2:
        print("[ERROR] Could not load recognizer model.")
        print("        First error:", e1)
        print("        Fallback error:", e2)
        print("[HINT] Run 'train_model.bat' to (re)train the model, then run again.")
        raise SystemExit(1)

try:
    le = pickle.loads(open("output/le.pickle", "rb").read())
except Exception as e:
    print("[ERROR] Could not load label encoder:", e)
    print("[HINT] Run 'train_model.bat' to (re)train the model, then run again.")
    raise SystemExit(1)

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# prepare notification state and directories
ensure_dir(UNKNOWN_CLIPS_DIR)
last_alert_time = 0.0

# notify admin that monitoring has started
try:
    send_text_message("Started monitoring")
except Exception as _e:
    print(f"[Telegram] Could not send start message: {_e}")

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

        # Flag to check if we need to record a clip
    record_clip_flag = False
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Optionally expand bounding box for better context
            if BBOX_EXPAND_RATIO > 0:
                bw = endX - startX
                bh = endY - startY
                expand_w = int(bw * BBOX_EXPAND_RATIO)
                expand_h = int(bh * BBOX_EXPAND_RATIO)
                startX = max(0, startX - expand_w)
                startY = max(0, startY - expand_h)
                endX = min(w - 1, endX + expand_w)
                endY = min(h - 1, endY + expand_h)

            # extract and enhance the face ROI
            face = frame[startY:endY, startX:endX]
            face = enhance_face_roi(face)
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = float(preds[j])
            name = le.classes_[j]
            # compute second-best probability for margin check
            sorted_probs = np.sort(preds)[::-1]
            second = float(sorted_probs[1]) if sorted_probs.shape[0] > 1 else 0.0
            margin = proba - second

            # color rule: green for known, red for unknown
            is_unknown = (proba < UNKNOWN_THRESHOLD) or (margin < MARGIN_MIN) or (proba < ABS_MIN_PROB)
            display_name = "Unknown" if is_unknown else name
            color = (0, 0, 255) if is_unknown else (0, 255, 0)

            # Check if this is an unknown person and we should record a clip
            current_time = time.time()
            if proba < UNKNOWN_THRESHOLD and (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
                record_clip_flag = True
                last_alert_time = current_time
                
                # Save the current frame as the first frame of the clip
                clip_path = record_clip(vs, frame.copy(), UNKNOWN_CLIP_SECONDS, UNKNOWN_CLIPS_DIR)
                
                # Get current date and time
                from datetime import datetime
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                confidence = f"{proba*100:.2f}%"
                
                # Format the message
                message = (
                    f"🚨 Unknown Person Detected!\n"
                    f"📅 Date: {now.strftime('%Y-%m-%d')}\n"
                    f"⏰ Time: {now.strftime('%H:%M:%S')}\n"
                    f"🔍 Confidence: {confidence}"
                )
                
                # Send the notification via Telegram
                try:
                    # Send text message with details
                    send_text_message(message)
                    
                    # Send the video clip if it exists
                    if os.path.exists(clip_path):
                        ok = send_video(clip_path, caption=f"Unknown person detected at {timestamp}")
                        if ok:
                            print(f"[ALERT] Sent unknown clip: {clip_path}")
                        else:
                            print(f"[ALERT] Failed to send clip via Telegram: {clip_path}")
                except Exception as e:
                    print(f"[ALERT] Error handling unknown detection: {e}")
                finally:
                    last_alert_time = time.time()

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()

# notify admin that monitoring has stopped
try:
    send_text_message("Stopped monitoring")
except Exception as _e:
    print(f"[Telegram] Could not send stop message: {_e}")