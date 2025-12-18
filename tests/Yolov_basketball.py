import cv2
from ultralytics import YOLO
import torch

# Load the YOLO11 model
model = YOLO("yolov/yolo11n-pose.pt")

# Open the video file
video_path = "Basketball_51_dataset/ft1/ft1_v108_004003_x264.mp4"
video_path = "1080p_Mehmet_demo_video.mp4"
cap = cv2.VideoCapture(video_path)

# --- 1. Get video properties and initialize VideoWriter ---
# Get video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
# 'mp4v' is a common and compatible codec for MP4 files.
# Choose an appropriate output path and filename
output_path = "output_video_with_annotations_mehmet.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Saving video to: {output_path}")

# --- 2. Loop through the video frames ---
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        # The 'verbose=False' argument suppresses the output for each frame, making the terminal cleaner.
        results = model.track(frame, persist=True, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # --- Write the annotated frame to the output video file ---
        out.write(annotated_frame)


        # Display the annotated frame (optional)
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# --- 3. Release resources ---
# Release the video capture object
cap.release()
# Release the video writer object
out.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

print("Video processing and saving complete.")