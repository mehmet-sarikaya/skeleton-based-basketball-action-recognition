import cv2
from ultralytics import YOLO
import torch
import uuid
import os
import sys

# --- 1. Load Model with Check ---
model_path = "yolov/yolo26n-pose.pt"

try:
    model = YOLO(model_path)
    print("✓ YOLO Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"FATAL: Could not initialize YOLO model: {e}")

# --- 2. Open Video with Check ---
video_path = "../videos/dribbling/Dribbling 1.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"FATAL: Video file not found at {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"FATAL: OpenCV could not open video file {video_path}. Check codecs or file integrity.")

print(f"✓ Video opened: {video_path}")

# --- 3. Initialize VideoWriter with Logic Checks ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Safety check for zero-property videos (common with corrupted headers)
if frame_width == 0 or frame_height == 0:
    raise ValueError("FATAL: Video frame dimensions are 0. The file may be corrupted.")

output_path = f"annotated_video_{str(uuid.uuid4())[:5]}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("ERROR: VideoWriter failed to open. Check write permissions or codec support.")
    cap.release()
    sys.exit(1)

print(f"✓ Saving annotated video to: {output_path}")

# --- 4. Processing Loop ---
try:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("End of video stream reached.")
            break

        # Run tracking
        results = model.track(
            source="https://www.youtube.com/watch?v=qXU5S0ywN40",     # oder 0 für Webcam
            persist=True,           # IDs über Frames behalten
            tracker="botsort.yaml", # oder "bytetrack.yaml"
            show=True
        )

        # Safety: check if results were actually returned
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            print("Anzahl der Leute:",len(results))
            print(results[0].keypoints.xyn)

            cv2.imshow("YOLO11 Tracking", annotated_frame)
        else:
            # If tracking fails for a single frame, just write the original
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("User interrupted processing.")
            break

except Exception as e:
    print(f"AN ERROR OCCURRED DURING PROCESSING: {e}")

finally:
    # --- 5. Clean Resource Release ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Resources released.")

print("Process finished.")