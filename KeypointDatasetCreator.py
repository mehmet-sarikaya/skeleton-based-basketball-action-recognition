import os.path
from collections import defaultdict

from VideoProcessor import VideoProcessor
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class KeypointDatasetCreator:
    def __init__(self, target_fps):
        self.keypoints_buffer = []
        self.model = YOLO("yolov/yolo26x-pose.pt")
        self.video_processor = VideoProcessor(target_fps=target_fps)
        self.target_fps = target_fps
        self.allowed_video_formats = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".flv", ".m4v"]

        self.annotation_dict = None

        self.data_dict = {
            "x": [],
            "y": [],
            "video_id": [],
            "is_original_data": []
        }

    def extract_keypoints_one_person_single_video(self, video_path: str, show=False):
        self.keypoints_buffer = []
        results = self.model.track(source=video_path, persist=True, show=show, verbose=False, stream=True)
        for result in results:
            self.extract_keypoint_coordinates(result)

    def extract_keypoint_coordinates(self, result):
        if result.keypoints is None or len(result.keypoints.xyn) == 0:
            self.keypoints_buffer.append(np.zeros((12, 3)))
            return

        boxes = result.boxes.xyxyn.cpu().numpy()

        # --- größte Box wählen ---
        best_idx = 0
        if len(boxes) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = np.argmax(areas)

        # --- Bounding Box ---
        x_min, y_min, x_max, y_max = boxes[best_idx]
        box_w = x_max - x_min
        box_h = y_max - y_min

        # Sicherheitscheck (sollte selten passieren)
        if box_w < 1e-6 or box_h < 1e-6:
            self.keypoints_buffer.append(np.zeros((12, 3)))
            return

        # --- Keypoints (bildnormalisiert) ---
        coords = result.keypoints.xyn[best_idx][5:].cpu().numpy()  # (12, 2)
        conf = result.keypoints.conf[best_idx][5:].cpu().numpy().reshape(-1, 1)

        # --- Bounding-Box-Normalisierung ---
        coords_rel = np.empty_like(coords)
        coords_rel[:, 0] = (coords[:, 0] - x_min) / box_w
        coords_rel[:, 1] = (coords[:, 1] - y_min) / box_h

        # Optional: clamp gegen numerische Ausreißer
        coords_rel = np.clip(coords_rel, 0.0, 1.0)

        # --- (x_rel, y_rel, conf) ---
        combined = np.concatenate([coords_rel, conf], axis=1)

        self.keypoints_buffer.append(combined)

    def create_keypoints_dataset(self, path, filter_videos=False):
        print(f"Processing {path}")
        path = Path(path)

        self.load_annotation_dict(path)

        video_files = [
            f for f in path.iterdir()
            if f.suffix.lower() in self.allowed_video_formats
        ]

        if filter_videos:
            video_files = self.filter_video_files(
                video_files,
                self.annotation_dict,
                exclude_flipped=True,  # oder False
                max_per_label=1000  # oder None
            )

        for video_path in tqdm(video_files, desc=f"Folder: {path.name}", unit="file", leave=False):
            self.extract_keypoints_one_person_single_video(str(video_path))
            self.append_to_dict_single_video(video_path)

        self.save_complete_dataset()

    def append_to_dict_single_video(self, path):
        filename = Path(path).stem

        try:
            self.data_dict["y"].append(self.annotation_dict[filename])
        except KeyError:
            print(f"Video_id {filename} not found in annotation_dict")
            self.keypoints_buffer = []
            return

        self.data_dict["x"].append(self.keypoints_buffer)
        self.keypoints_buffer = []

        video_id = filename.split("_")[0]
        self.data_dict["video_id"].append(video_id)

        self.data_dict["is_original_data"].append("flipped" not in filename)

    def get_padded_buffer(self):
        # Handle length to ensure exactly 16 frames
        if len(self.keypoints_buffer) > 16:
            # Truncate: take the first 16 frames
            processed_buffer = self.keypoints_buffer[:16]
        elif len(self.keypoints_buffer) < 16:
            # Padding: fill with zeros if shorter than 16
            padding_size = 16 - len(self.keypoints_buffer)
            padding = [np.zeros((12, 3)) for _ in range(padding_size)]
            processed_buffer = self.keypoints_buffer + padding
        else:
            processed_buffer = self.keypoints_buffer

        return processed_buffer

    def save_complete_dataset(self, output_path="dataset_all.npz"):
        x_array = np.array(self.data_dict["x"], dtype="float32")

        save_dict = {
            "x": x_array,
            "y": np.array(self.data_dict["y"]),
            "video_id": np.array(self.data_dict["video_id"]),
            "is_original_data": np.array(self.data_dict["is_original_data"])
        }

        np.savez(output_path, **save_dict)
        print(f"Dataset gespeichert. Shape von x: {x_array.shape}")

    def load_annotation_dict(self, path: Path):
        dict_path = path / "annotation_dict.json"

        if not dict_path.exists():
            dict_path = path.parent / "annotation_dict.json"

        if not dict_path.exists():
            raise FileNotFoundError(f"annotation_dict.json wurde weder in {path} noch in {path.parent} gefunden!")

        with dict_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # keys: video_id (string), values: label_id (int)
        self.annotation_dict = {str(k): int(v) for k, v in data.items()}

    def save_keypoint_single_video(self, path):
        filename = Path(path).stem
        folder_path = Path(path).parent
        out_file_name = filename + ".npz"

        np.savez_compressed(os.path.join(folder_path, out_file_name), data=self.keypoints_buffer)

        self.keypoints_buffer = []

    def filter_video_files(
            self,
            video_files,
            annotation_dict,
            exclude_flipped: bool = True,
            max_per_label: int | None = None
    ):
        label_counts = defaultdict(int)
        filtered = []

        for vp in video_files:
            name = vp.stem

            # Option 1: flipped ausschließen
            if exclude_flipped and "_flipped" in name:
                continue

            # Label aus Annotation Dict
            label = annotation_dict.get(name)
            if label is None:
                continue

            # Option 2: max Samples pro Label
            if max_per_label is not None:
                if label_counts[label] >= max_per_label:
                    continue
                label_counts[label] += 1

            filtered.append(vp)

        return filtered
