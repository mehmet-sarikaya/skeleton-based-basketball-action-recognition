import os.path

from VideoProcessor import VideoProcessor
from ultralytics import YOLO
import numpy as np
from pathlib import Path


class DatasetCreator:
    def __init__(self):
        self.keypoints_buffer = []
        self.model = YOLO("yolov/yolo11n-pose.pt")  # Load the YOLO11 Pose Detection model

    def save_single_person_pose_array_no_face_single_video(self, video_path: str, target_fps=10):
        video_processor = VideoProcessor()
        video_processor.ProcessVideoPerFrame(video_path, self.extract_keypoint_coordinates, target_fps=target_fps)

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
        self.save_single_person_pose_array_no_face_single_video(path)

        filename = Path(path).stem
        folder_path = Path(path).parent
        out_file_name = filename + "_keypoints.npz"

        np.savez_compressed(os.path.join(folder_path, out_file_name), data=self.keypoints_buffer)


creator = DatasetCreator()
creator.create_keypoints_dataset("videos/1080p_Mehmet_demo_video.mp4")
