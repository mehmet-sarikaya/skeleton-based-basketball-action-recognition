from VideoProcessor import VideoProcessor
from ultralytics import YOLO
from collections import deque
import tensorflow as tf
import numpy as np
import cv2

class ModelTester:
    def __init__(self, app):
        self.keypoints_buffer = []
        self.buffer_counter = 0

        self.label_id_dic = app.label_id_dic
        self.id_to_label = {v: k for k, v in self.label_id_dic.items()}

        self.yolo_model = YOLO("yolov/yolo26x-pose.pt")  # Load the YOLO11 Pose Detection model
        self.video_processor = VideoProcessor(target_fps=app.fps)
        self.model_path = None

        self.target_fps = app.fps
        self.win_len_sec = app.win_len_sec
        self.stride_len_sec = 1.0
        self.stride_num_frames = app.stride_len_sec * app.fps

        self.max_len = int(self.target_fps * self.win_len_sec)

        self.basketball_model = None
        self.current_class_id = 0
        self.pred_conf = 0
        self.current_frame = None
        self.current_annotated_frame = None

        self.allowed_video_formats = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".flv", ".m4v"]

    def test_model_on_video(self, video_path: str, model_path:str):
        self.keypoints_buffer = deque(maxlen=self.max_len)
        self.model_path = model_path

        self.video_processor.process_video_per_frame_at_constant_fps(
            video_path, frame_callback=self.collect_frames_and_test_model, show_frames=True)

        """
        self.video_processor.process_camera_per_frame_at_constant_fps(
            frame_callback=self.collect_frames_and_test_model, show_frames=True)
        """

    def collect_frames_and_test_model(self, frame):
        # extract Keypoint Coordinates (returns just one result since one picture)
        self.current_frame = frame
        results = self.yolo_model.track(frame, persist=True, verbose=False)
        result = results[0]

        if results and len(results) > 0:
            annotated_frame = results[0].plot()
            self.draw_classification(result, annotated_frame)
            cv2.imshow("YOLO11 Tracking", annotated_frame)

        if result is None or len(result.keypoints.xyn) == 0:
            return

        xyn = result.keypoints.xyn  # normalized keypoints of all persons

        # if multiple persons are there extract just from single person
        # and without head joints
        single_person_coordinates = xyn[0][5:].cpu().numpy()

        boxes = result.boxes.xyxyn.cpu().numpy()


        # --- Bounding Box ---
        x_min, y_min, x_max, y_max = boxes[0]
        box_w = x_max - x_min
        box_h = y_max - y_min

        # --- Keypoints (bildnormalisiert) ---
        coords = result.keypoints.xyn[0][5:].cpu().numpy()  # (12, 2)
        conf = result.keypoints.conf[0][5:].cpu().numpy().reshape(-1, 1)

        # --- Bounding-Box-Normalisierung ---
        coords_rel = np.empty_like(coords)
        coords_rel[:, 0] = (coords[:, 0] - x_min) / box_w
        coords_rel[:, 1] = (coords[:, 1] - y_min) / box_h

        # Optional: clamp gegen numerische Ausreißer
        coords_rel = np.clip(coords_rel, 0.0, 1.0)

        self.keypoints_buffer.append(coords_rel)

        self.recognize_activity()

    def recognize_activity(self):
        if self.basketball_model is None:
            self.basketball_model = tf.keras.models.load_model(self.model_path)

        self.buffer_counter += 1

        if len(self.keypoints_buffer) >= self.max_len and self.buffer_counter % self.stride_num_frames == 0:
            self.buffer_counter = 0
            window = np.asarray(self.keypoints_buffer, dtype=np.float32)
            x = window[None, ..., None]

            probs = self.basketball_model.predict(x, verbose=0)
            self.current_class_id = int(np.argmax(probs, axis=-1)[0])
            self.pred_conf = float(np.max(probs))

            predicted_label = self.label_id_dic[self.current_class_id]
            print(f"Predicted Label: {predicted_label}, Confidence: {self.pred_conf} ")

    def draw_classification(self,result, annotated_frame):
        if result.boxes is not None and len(result.boxes) > 0:
            # bei 1 Person einfach die erste Box
            x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy().astype(int)

            text = f"{self.label_id_dic[self.current_class_id]} ({self.pred_conf:.2f})"

            # 4) Text über der Box platzieren (mit kleiner Hintergrundbox für Lesbarkeit)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y_text = max(y1 - 10, th + 10)

            # Hintergrund
            cv2.rectangle(
                annotated_frame,
                (x1, y_text - th - baseline),
                (x1 + tw, y_text + baseline),
                (0, 0, 0),
                thickness=-1
            )

            # Text
            cv2.putText(
                annotated_frame,
                text,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
