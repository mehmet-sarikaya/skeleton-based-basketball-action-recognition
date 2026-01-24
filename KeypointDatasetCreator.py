import os.path
from VideoProcessor import VideoProcessor
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from tqdm import tqdm

class KeypointDatasetCreator:
    def __init__(self, target_fps):
        self.keypoints_buffer = []
        self.model = YOLO("yolov/yolo26x-pose.pt")  # Load the YOLO11 Pose Detection model
        self.video_processor = VideoProcessor(target_fps=target_fps)
        self.target_fps = target_fps
        self.allowed_video_formats = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".flv", ".m4v"]

        self.annotation_dict = None

    def extract_keypoints_one_person_single_video(self, video_path: str):
        self.keypoints_buffer = []
        self.video_processor.process_video_per_frame_at_constant_fps(
            video_path, frame_callback=self.extract_keypoint_coordinates, verbose=False)

    def extract_keypoint_coordinates(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.keypoints.xyn) == 0:
            # Falls kein Skelett gefunden wurde: Buffer mit Nullen füllen
            # oder Frame überspringen (Nullen halten die Sequenzlänge konstant)
            self.keypoints_buffer.append(np.zeros((12, 2)))
            return

        coords = result.keypoints.xyn[0][5:].cpu().numpy()
        conf = result.keypoints.conf[0][5:].cpu().numpy()

        # Setze unsichere Gelenke auf (0,0)
        threshold = 0.5
        coords[conf < threshold] = [0, 0]

        self.keypoints_buffer.append(coords)

    import os
    from pathlib import Path
    from tqdm import tqdm

    def create_keypoints_dataset(self, path):
        print(f"Processing {path}")
        sub_dirs = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]

        for sub in tqdm(sub_dirs, desc="Overall Progress (Folders)", unit="folder"):
            sub_dir_path = os.path.join(path, sub)

            video_files = [
                f for f in os.listdir(sub_dir_path)
                if Path(f).suffix in self.allowed_video_formats
            ]

            for filename in tqdm(video_files, desc=f"Folder: {sub}", unit="file", leave=False):
                file_path = os.path.join(sub_dir_path, filename)

                self.extract_keypoints_one_person_single_video(file_path)
                self.save_keypoint_single_video(file_path)

    def save_keypoint_single_video(self, path):
        filename = Path(path).stem
        folder_path = Path(path).parent
        out_file_name = filename + ".npz"

        np.savez_compressed(os.path.join(folder_path, out_file_name), data=self.keypoints_buffer)

        self.keypoints_buffer = []
