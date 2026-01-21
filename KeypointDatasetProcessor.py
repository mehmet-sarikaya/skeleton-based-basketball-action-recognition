import os
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Augmenter import Augmenter
import json
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d


class KeypointDatasetProcessor:
    def __init__(self, fps, label_id_dic):
        # chosen dynamically
        self.current_label_keypoints_per_video = None
        self.win_len_sec = None
        self.stride_len_sec = None

        self.label_id_dic = label_id_dic

        self.data_dic = {
            i: []
            for i in range(11)
        }
        self.annotation_dict = None

        self.x = []
        self.y = []
        self.subjects = []
        self.is_original_data = []

        self.allowed_datatypes = [".npy", ".npz"]
        self.fps = fps

        self.augmenter = Augmenter(self.fps)

    def load_all_keypoints_data_in_dir_old(self, path):
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

    def load_annotation_dict(self, path: Path):
        dict_path = path / "annotation_dict.json"
        with dict_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # keys: video_id (string), values: label_id (int)
        self.annotation_dict = {str(k): int(v) for k, v in data.items()}

    def load_all_keypoints_data_in_dir(self, path):
        if Path("dataset_all.npz").exists():
            self.load_big_file()
            return

        path = Path(path)
        self.load_annotation_dict(path)
        global_subject_counter = 0
        for video_id, label_id in tqdm(self.annotation_dict.items(), desc="Loading keypoints"):
            video_id = video_id + ".npz"
            file_path = path / video_id

            data = np.load(file_path, allow_pickle=True)["data"]
            data = np.array(data)
            data = self.smooth_and_centre_data(data)
            self.x.append(data)
            self.y.append(label_id)

            #rather video_id_belonging
            self.subjects.append(global_subject_counter)

            if "_flipped" in video_id:
                self.is_original_data.append(False)
            else:
                global_subject_counter += 1
                self.is_original_data.append(True)

        self.x = pad_sequences(
            self.x,
            maxlen=15,
            dtype="float32",
            padding="post",  # pad at end
            truncating="post",  # cut at end if longer
            value=0.0
        )
        self.x = np.expand_dims(self.x, axis=-1)

        self.y = np.array(self.y)
        self.subjects = np.array(self.subjects)
        self.is_original_data = np.array(self.is_original_data)

        np.savez_compressed(
            "dataset_all.npz",
            X=self.x,
            y=self.y,
            subjects=self.subjects,
            is_original=self.is_original_data
        )

    def load_big_file(self):
        data = np.load("dataset_all.npz")
        self.x = data["X"]
        self.y = data["y"]
        self.subjects = data["subjects"]
        self.is_original_data = data["is_original"]

        for i in tqdm(range(len(self.x))):
            transformed = self.smooth_and_centre_data(self.x[i])
            self.x[i] = np.expand_dims(transformed, axis=-1)

    def smooth_and_centre_data(self, sequence):
        # 1. Sicherstellen, dass es ein NumPy Array ist
        sequence = np.array(sequence)

        # Falls sequence (15, 12, 2, 1) ist, auf (15, 12, 2) bringen
        if sequence.ndim == 4:
            sequence = np.squeeze(sequence, axis=-1)

        # 2. Zeitliche Glättung
        sequence = uniform_filter1d(sequence, size=3, axis=0)

        # 3. Relative Zentrierung auf die Hüft-Mitte
        for i in range(len(sequence)):
            # Wir erzwingen hier (12, 2), falls noch eine Dimension dran klebt
            frame_coords = sequence[i].reshape(12, 2)

            hip_l = frame_coords[6]
            hip_r = frame_coords[7]

            if not (np.all(hip_l == 0) and np.all(hip_r == 0)):
                center = (hip_l + hip_r) / 2

                # Maske: Welche der 12 Gelenke sind erkannt worden?
                # mask.shape wird (12,)
                mask = np.any(frame_coords != 0, axis=1)

                # Der eigentliche Fix:
                # Wir subtrahieren den Center-Punkt nur von den validen Gelenken
                frame_coords[mask] = frame_coords[mask] - center

                # Zurückschreiben in die sequence
                sequence[i] = frame_coords

        return sequence

    def load_keypoints_data(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)

            if not os.path.isfile(file_path):
                # print("skip ", file_path, " this is no file")
                continue

            if not Path(file_path).suffix in self.allowed_datatypes:
                # print(f"skip {file_path} must be of type: {self.allowed_datatypes}")
                continue

            # print(f"Loading {file_path}")
            file_keypoints = np.load(file_path)["data"]
            # print(file_keypoints)

            self.current_label_keypoints_per_video.append(file_keypoints)

    def prepare_data_for_training_old(self, win_len_sec=3, stride_len_sec=1):
        # definitions for training
        self.win_len_sec = win_len_sec
        win_len = int(win_len_sec * self.fps)

        self.stride_len_sec = stride_len_sec
        stride_len = int(stride_len_sec * self.fps)

        global_subject_counter = 0

        for gesture_str, data_list in self.data_dic.items():
            for keypoint_data in data_list:
                data_len = len(keypoint_data)
                # TODO: allow last window to be not at the length of win_len -> padding
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
                # TODO: allow last window to be not at the length of win_len -> padding
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
