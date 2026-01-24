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

    def extract_keypoints_one_person_single_video(self, video_path: str, show=False):
        self.keypoints_buffer = []
        results = self.model.track(source=video_path, persist=True, show=show, verbose=False, stream=True)
        for result in results:
            self.extract_keypoint_coordinates(result)

    def extract_keypoint_coordinates(self, result):
        if result.keypoints is None or len(result.keypoints.xyn) == 0:
            self.keypoints_buffer.append(np.zeros((12, 3)))
            return

        # print(result)

        # Find the index of the person closest to the center
        # result.boxes.xyxyn contains normalized [xmin, ymin, xmax, ymax]
        boxes = result.boxes.xyxyn.cpu().numpy()

        best_idx = 0
        if len(boxes) > 1:
            max_area = -1.0
            for i, box in enumerate(boxes):
                # Fläche berechnen: (xmax - xmin) * (ymax - ymin)
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height

                if area > max_area:
                    max_area = area
                    best_idx = i

        if result.boxes.id is not None:
            track_id = int(result.boxes.id[best_idx].item())
            # print(f"Tracking ID in center: {track_id}")
        else:
            # print("No Tracker ID assigned yet.")
            pass

        coords = result.keypoints.xyn[best_idx][5:].cpu().numpy()
        conf = result.keypoints.conf[best_idx][5:].cpu().numpy()
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
