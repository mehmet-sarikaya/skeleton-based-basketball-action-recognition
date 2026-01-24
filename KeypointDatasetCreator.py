import os.path
from VideoProcessor import VideoProcessor
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class KeypointDatasetCreator:
    def __init__(self, target_fps):
        self.keypoints_buffer = []
        self.model = YOLO("yolov/yolo26x-pose.pt")  # Load the YOLO11 Pose Detection model
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

    def extract_keypoints_one_person_single_video(self, video_path: str):
        self.keypoints_buffer = []
        self.video_processor.process_video_per_frame_at_constant_fps(
            video_path, frame_callback=self.extract_keypoint_coordinates, verbose=False)

    def extract_keypoint_coordinates(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.keypoints.xyn) == 0:
            self.keypoints_buffer.append(np.zeros((12, 2)))
            return

        coords = result.keypoints.xyn[0][5:].cpu().numpy()
        conf = result.keypoints.conf[0][5:].cpu().numpy()
        conf = conf.reshape(-1, 1)

        combined = np.concatenate([coords, conf], axis=1)

        self.keypoints_buffer.append(combined)

    def create_keypoints_dataset(self, path):
        print(f"Processing {path}")
        path = Path(path)

        self.load_annotation_dict(path)

        video_files = [
            f for f in path.iterdir()
            if f.suffix.lower() in self.allowed_video_formats
        ]

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

        self.data_dict["is_original_data"].append("flipped" in filename)



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
