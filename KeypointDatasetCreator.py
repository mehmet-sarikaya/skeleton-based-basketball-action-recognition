import os.path

from VideoProcessor import VideoProcessor
from ultralytics import YOLO
import numpy as np
from pathlib import Path


class KeypointDatasetCreator:
    def __init__(self):
        self.keypoints_buffer = []
        self.model = YOLO("yolov/yolo11n-pose.pt")  # Load the YOLO11 Pose Detection model
        self.video_processor = VideoProcessor()

    def extract_keypoints_one_person_single_video(self, video_path: str, target_fps=10):
        self.keypoints_buffer = []
        self.video_processor.process_video_per_frame(video_path, self.extract_keypoint_coordinates, target_fps=target_fps)

    def extract_keypoint_coordinates(self, frame):
        # extract Keypoint Coordinates (returns just one result since one picture)
        results = self.model(frame, verbose=False)
        result = results[0]

        if result is None or len(result.keypoints.xyn) == 0:
            return

        xyn = result.keypoints.xyn  # normalized keypoints of all persons

        # if multiple persons are there extract just from single person
        single_person_coordinates = xyn[0].cpu().numpy()

        self.keypoints_buffer.append(single_person_coordinates)

    def create_keypoints_dataset(self, path):
        print(f"Processing {path}")

        for sub in os.listdir(path):
            sub_dir_path = os.path.join(path,sub)

            # don't process single file here
            if not os.path.isdir(sub_dir_path):
                continue

            for filename in os.listdir(sub_dir_path):
                print(f"---------------- Processing {filename} ----------------")
                file_path = os.path.join(sub_dir_path, filename)

                self.extract_keypoints_one_person_single_video(file_path)
                self.save_keypoint_single_video(file_path)

    def save_keypoint_single_video(self, path):
        filename = Path(path).stem
        folder_path = Path(path).parent
        out_file_name = filename + "_keypoints.npz"

        np.savez_compressed(os.path.join(folder_path, out_file_name), data=self.keypoints_buffer)
        self.keypoints_buffer = []


if __name__ == "__main__":
    creator = KeypointDatasetCreator()
    creator.create_keypoints_dataset("videos/")
