import os
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Augmenter import Augmenter


class KeypointDatasetProcessor:
    def __init__(self, fps, label_id_dic):
        # chosen dynamically
        self.current_label_keypoints_per_video = None
        self.win_len_sec = None
        self.stride_len_sec = None

        self.label_id_dic = label_id_dic

        self.data_dic = {
            "dribbling": [],
            "passing": [],
            "throw": []
        }

        self.x = []
        self.y = []
        self.subjects = []
        self.is_original_data = []

        self.allowed_datatypes = [".npy", ".npz"]
        self.fps = fps

        self.augmenter = Augmenter(self.fps)

    def load_all_keypoints_data_in_dir(self, path):
        for subdir in os.listdir(path):
            subdir_path = os.path.join(path,subdir)

            # don't process single file here
            if not os.path.isdir(subdir_path):
                continue

            if subdir in self.data_dic:
                self.current_label_keypoints_per_video = self.data_dic[subdir]
                self.load_keypoints_data(subdir_path)
            else:
                print(f"Don't process {subdir} folder since it's not named after any label: {self.data_dic.keys()}")

    def load_keypoints_data(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)

            if not os.path.isfile(file_path):
                # print("skip ", file_path, " this is no file")
                continue

            if not Path(file_path).suffix in self.allowed_datatypes:
                # print(f"skip {file_path} must be of type: {self.allowed_datatypes}")
                continue

            print(f"Loading {file_path}")
            file_keypoints = np.load(file_path)["data"]
            print(file_keypoints)

            self.current_label_keypoints_per_video.append(file_keypoints)

    def prepare_data_for_training(self, win_len_sec=3, stride_len_sec=1):
        # definitions for training
        self.win_len_sec = win_len_sec
        win_len = int(win_len_sec * self.fps)

        self.stride_len_sec = stride_len_sec
        stride_len = int(stride_len_sec * self.fps)

        global_subject_counter = 0

        for gesture_str, data_list in self.data_dic.items():
            for keypoint_data in data_list:
                data_len = len(keypoint_data)
                for start_pos_window in range(0, data_len - win_len + 1, stride_len):
                    end_pos_window = start_pos_window + win_len

                    # slicing np array and adding to x
                    new_sliding_window = keypoint_data[start_pos_window:end_pos_window]
                    new_sliding_window = np.array(new_sliding_window)

                    gesture_id = self.label_id_dic[gesture_str]

                    self.x.append(new_sliding_window)
                    self.y.append(gesture_id)
                    self.subjects.append(global_subject_counter)
                    self.is_original_data.append(True)

                    # augmentation
                    augmented_windows = self.augmenter.augment_window(new_sliding_window)
                    self.x.extend(augmented_windows)
                    for i in range(len(augmented_windows)):
                        self.y.append(gesture_id)
                        self.subjects.append(global_subject_counter)
                        self.is_original_data.append(False)

                global_subject_counter += 1

        self.x = pad_sequences(
            self.x,
            maxlen=win_len,
            dtype='float32',
            padding='post',  # Nullen AM ENDE anfügen
            value=0.0
        )

        self.x = np.expand_dims(self.x, axis=-1)
        self.y = np.array(self.y)
        self.subjects = np.array(self.subjects)
        self.is_original_data = np.array(self.is_original_data)

    def print_data_after_preparation(self):
        print("\n" + "=" * 30)
        print("DATA PREPARATION SUMMARY")
        print("=" * 30)
        print(f"Total Windows (Samples): {len(self.x)}")
        print(f"Window Shape (Frames x Features): {self.x.shape[1:] if len(self.x) > 0 else 'N/A'}")
        print(f"Total Unique Subjects: {len(np.unique(self.subjects))}")

        print("\nWindows per Gesture:")
        # Reverse lookup for gesture names
        id_to_label = {v: k for k, v in self.label_id_dic.items()}
        unique_labels, counts = np.unique(self.y, return_counts=True)

        for label_id, count in zip(unique_labels, counts):
            print(f" - {id_to_label[label_id]}: {count} windows")
        print("=" * 30 + "\n")


if __name__ == "__main__":
    data_processor = KeypointDatasetProcessor()
    data_processor.load_all_keypoints_data_in_dir("videos")
