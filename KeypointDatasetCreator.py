import os.path
from VideoProcessor import VideoProcessor
from ultralytics import YOLO
import numpy as np
from pathlib import Path

class KeypointDatasetCreator:
    def __init__(self, target_fps):
        self.keypoints_buffer = []
        self.model = YOLO("yolov/yolo26n-pose.pt")  # Load the YOLO11 Pose Detection model
        self.video_processor = VideoProcessor(target_fps=target_fps)
        self.target_fps = target_fps
        self.allowed_video_formats = [".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".flv", ".m4v"]

        self.annotation_dict = None

    def extract_keypoints_one_person_single_video(self, video_path: str):
        self.keypoints_buffer = []
        self.video_processor.process_video_per_frame_at_constant_fps(video_path, frame_callback=self.extract_keypoint_coordinates)

    def extract_keypoint_coordinates(self, frame):
        # extract Keypoint Coordinates (returns just one result since one picture)
        results = self.model.track(frame, persist=True, verbose=False)
        result = results[0]

        if result is None or len(result.keypoints.xyn) == 0:
            return

        xyn = result.keypoints.xyn  # normalized keypoints of all persons

        # if multiple persons are there extract just from single person
        # and without head joints
        single_person_coordinates = xyn[0][5:].cpu().numpy()

        self.keypoints_buffer.append(single_person_coordinates)

    def create_keypoints_dataset(self, path):
        print(f"Processing {path}")

        for sub in os.listdir(path):
            sub_dir_path = os.path.join(path,sub)

            # don't process single file here
            if not os.path.isdir(sub_dir_path):
                continue

            for filename in os.listdir(sub_dir_path):
                # process only videos
                print(Path(filename).suffix)
                if not Path(filename).suffix in self.allowed_video_formats:
                    continue

                print(f"---------------- Processing {filename} ----------------")
                file_path = os.path.join(sub_dir_path, filename)

                self.extract_keypoints_one_person_single_video(file_path)
                self.save_keypoint_single_video(file_path)

    def save_keypoint_single_video(self, path):
        filename = Path(path).stem
        folder_path = Path(path).parent
        out_file_name = filename + ".npz"

        np.savez_compressed(os.path.join(folder_path, out_file_name), data=self.keypoints_buffer)

        self.keypoints_buffer = []
