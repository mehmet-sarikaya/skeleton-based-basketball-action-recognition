from Graph_Model_Creator import STGCNBlock, GraphConv
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

        self.yolo_model = YOLO("yolov/yolo26n-pose.pt")  # Load the YOLO11 Pose Detection model
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

        self.model_name = app.model_name

        self.allowed_video_formats = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".flv", ".m4v"]

        self.STGCN_CUSTOM_OBJECTS = {
            "STGCNBlock": STGCNBlock,
            "GraphConv": GraphConv
        }

        self.video_writer = None

    def test_model_on_video(self, video_path: str, model_path:str, save_video=False):
        self.keypoints_buffer = deque(maxlen=self.max_len)
        self.model_path = model_path

        if save_video:
            # Output-Name generieren
            output_path = video_path.rsplit('.', 1)[0] + "_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # VideoProcessor muss uns hier kurz helfen, die FPS und Größe zu wissen
            # Wir initialisieren den Writer final im ersten Frame-Callback
            self.output_path = output_path
            self.fourcc = fourcc

        self.video_processor.process_video_per_frame_at_constant_fps(
            video_path, frame_callback=self.collect_frames_and_test_model, show_frames=True)

    def test_model_on_camera(self, model_path:str, camera_id_or_url):
        self.keypoints_buffer = deque(maxlen=self.max_len)
        self.model_path = model_path

        self.video_processor.process_camera_per_frame_at_constant_fps(
            frame_callback=self.collect_frames_and_test_model, camera_id_or_url=camera_id_or_url, show_frames=True)


    def collect_frames_and_test_model(self, frame):
        # extract Keypoint Coordinates (returns just one result since one picture)
        self.current_frame = frame
        results = self.yolo_model.track(frame, persist=True, verbose=False)
        result = results[0]

        if results and len(results) > 0:
            annotated_frame = results[0].plot()
            self.draw_classification(result, annotated_frame)

            if hasattr(self, 'output_path'):
                if self.video_writer is None:
                    h, w = annotated_frame.shape[:2]
                    self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.target_fps, (w, h))
                self.video_writer.write(annotated_frame)

            cv2.imshow("YOLO11 Tracking", annotated_frame)

        if result is None or len(result.keypoints.xyn) == 0:
            return

        boxes = result.boxes.xyxyn.cpu().numpy()  # (N, 4)
        kps = result.keypoints.xyn.cpu().numpy()  # (N, K, 2)

        # track ids (bei persist=True typischerweise vorhanden)
        ids = None
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)  # (N,)
        else:
            # fallback: wenn keine IDs da sind
            ids = np.arange(len(boxes), dtype=int)

        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = boxes[i]
            box_w = x_max - x_min
            box_h = y_max - y_min
            if box_w <= 1e-6 or box_h <= 1e-6:
                continue

            # Keypoints ohne Kopf (ab Index 5)
            coords = kps[i][5:]  # (12,2)

            coords_rel = np.empty_like(coords)
            coords_rel[:, 0] = (coords[:, 0] - x_min) / box_w
            coords_rel[:, 1] = (coords[:, 1] - y_min) / box_h
            coords_rel = np.clip(coords_rel, 0.0, 1.0)

            track_id = int(ids[i])

            # pro Person eigener Buffer
            if not hasattr(self, "keypoints_buffers"):
                self.keypoints_buffers = {}
            if track_id not in self.keypoints_buffers:
                self.keypoints_buffers[track_id] = deque(maxlen=self.max_len)

            self.keypoints_buffers[track_id].append(coords_rel)

            # pro Person klassifizieren
            self.recognize_activity(track_id)

    def recognize_activity(self, track_id: int):
        if self.basketball_model is None:
            try:
                self.basketball_model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=self.STGCN_CUSTOM_OBJECTS
                )
            except Exception as e:
                self.basketball_model = tf.keras.models.load_model(self.model_path)

        # pro Person eigener Counter (sonst "shared stride" über alle)
        if not hasattr(self, "buffer_counters"):
            self.buffer_counters = {}
        if track_id not in self.buffer_counters:
            self.buffer_counters[track_id] = 0

        self.buffer_counters[track_id] += 1
        kp_buf = self.keypoints_buffers[track_id]

        if len(kp_buf) >= self.max_len and self.buffer_counters[track_id] % self.stride_num_frames == 0:
            self.buffer_counters[track_id] = 0

            window = np.asarray(kp_buf, dtype=np.float32)
            x = window[None, ..., None]

            probs = self.basketball_model.predict(x, verbose=0)

            # wenn du pro Person label/conf speichern willst:
            if not hasattr(self, "current_class_id_by_id"):
                self.current_class_id_by_id = {}
                self.pred_conf_by_id = {}

            class_id = int(np.argmax(probs, axis=-1)[0])
            conf = float(np.max(probs))

            self.current_class_id_by_id[track_id] = class_id
            self.pred_conf_by_id[track_id] = conf

            predicted_label = self.label_id_dic[class_id]
            print(f"[id={track_id}] Predicted Label: {predicted_label}, Confidence: {conf}")

    def draw_classification(self, result, annotated_frame):
        if result.boxes is None or len(result.boxes) == 0:
            return

        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)

        # track ids
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.arange(len(boxes_xyxy), dtype=int)

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            track_id = int(ids[i])

            # default falls noch keine Prediction für diese ID existiert
            class_id = None
            conf = 0.0

            if hasattr(self, "current_class_id_by_id") and track_id in self.current_class_id_by_id:
                class_id = self.current_class_id_by_id[track_id]
                conf = float(self.pred_conf_by_id.get(track_id, 0.0))
                label = self.label_id_dic[class_id]
                text = f"{label} ({conf:.2f})"
            else:
                text = f"id={track_id} (...)"

            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y_text = max(y1 - 10, th + 10)

            cv2.rectangle(
                annotated_frame,
                (x1, y_text - th - baseline),
                (x1 + tw, y_text + baseline),
                (0, 0, 0),
                thickness=-1
            )

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
