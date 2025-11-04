import os
import cv2
import imutils
from pathlib import Path
import argparse
from typing import Optional


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def find_available_cameras(max_index: int = 10):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.read()[0]:
            available.append(idx)
        if cap is not None:
            cap.release()
    return available


def select_camera(arg_cam: Optional[int]) -> int:
    if arg_cam is not None:
        return arg_cam
    env_cam = os.getenv("CAMERA_INDEX")
    if env_cam and env_cam.isdigit():
        return int(env_cam)
    cams = find_available_cameras(10)
    if len(cams) == 0:
        print("No cameras detected, defaulting to index 0.")
        return 0
    if len(cams) == 1:
        print(f"Using detected camera index: {cams[0]}")
        return cams[0]
    print(f"Available cameras: {cams}")
    try:
        return int(input("Select camera index: "))
    except Exception:
        print(f"Invalid input, defaulting to {cams[0]}")
        return cams[0]


def main():
    parser = argparse.ArgumentParser(description="Enroll a person's face by capturing images to the dataset.")
    parser.add_argument("--name", type=str, default=None, help="Person name for dataset folder (dataset/<name>)")
    parser.add_argument("--count", type=int, default=30, help="Number of images to capture")
    parser.add_argument("--camera", type=int, default=None, help="Camera index to use")
    args = parser.parse_args()

    person_name = args.name or input("Enter person name: ").strip()
    if not person_name:
        print("Name is required.")
        return

    save_dir = Path("dataset") / person_name
    ensure_dir(save_dir)

    cam_index = select_camera(args.camera)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Failed to open camera {cam_index}")
        return

    print("Instructions:\n - Press 'c' to capture a photo.\n - Press 'q' to quit early.")
    captured = 0
    next_index = 0
    # find next index to avoid overwrite
    while (save_dir / f"{next_index:05d}.png").exists():
        next_index += 1

    try:
        while captured < args.count:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = imutils.resize(frame, width=600)
            cv2.putText(frame, f"Name: {person_name}  Captured: {captured}/{args.count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Enroll Face", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                out_path = save_dir / f"{next_index:05d}.png"
                cv2.imwrite(str(out_path), frame)
                print(f"Saved {out_path}")
                captured += 1
                next_index += 1
        print(f"Enrollment finished. Total saved: {captured}")
    finally:
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    main()
