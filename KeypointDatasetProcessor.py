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
        self.video_id = []
        self.is_original_data = []

        self.allowed_datatypes = [".npy", ".npz"]
        self.fps = fps

        self.augmenter = Augmenter(self.fps)

    def load_dataset(self, path, smooth_data=False, include_conf=False, augment_data = True):
        path = Path(path) / "dataset_all.npz"
        with np.load(path, allow_pickle=True) as data:
            print(data)
            self.x = data["x"]
            self.y = data["y"]
            try:
                self.video_id = data["video_id"]
            except KeyError:
                self.video_id = data["subjects"]

            try:
                self.is_original_data = data["is_original_data"]
            except KeyError:
                self.is_original_data = data["is_original"]

        print(f"Daten geladen: {self.x.shape}")

        if not include_conf:
            self.drop_confidence_axis()

        if smooth_data:
            for i in range(len(self.x)):
                self.x[i] = self.smooth_and_centre_data(self.x[i])

        if augment_data:
            self.augment_data()

    def augment_data(self):
        X_aug = []
        y_aug = []
        vid_aug = []
        orig_aug = []

        for i in tqdm(range(len(self.x)), desc="Augmenting windows", unit="window"):
            window = self.x[i]  # (T,12,F)
            aug_windows = self.augmenter.augment_window(window)

            for w in aug_windows:
                X_aug.append(w)
                y_aug.append(self.y[i])
                vid_aug.append(self.video_id[i])
                orig_aug.append(False)

        # Original + Augmented zusammenführen
        self.x = np.concatenate([self.x, np.asarray(X_aug, dtype=self.x.dtype)], axis=0)
        self.y = np.concatenate([self.y, np.asarray(y_aug, dtype=self.y.dtype)], axis=0)
        self.video_id = np.concatenate([self.video_id, np.asarray(vid_aug, dtype=self.video_id.dtype)], axis=0)
        self.is_original_data = np.concatenate(
            [self.is_original_data, np.asarray(orig_aug, dtype=bool)],
            axis=0
        )

    def smooth_and_centre_data(self, sequence, center_data=True):
        sequence = np.array(sequence)

        if sequence.ndim == 4:
            sequence = np.squeeze(sequence, axis=-1)

        # Nur X und Y glätten (Index 0 und 1 auf der letzten Achse)
        # sequence[..., :2] wählt alle Frames und Joints, aber nur x, y
        sequence[..., :2] = uniform_filter1d(sequence[..., :2], size=3, axis=0)

        if not center_data:
            return sequence

        # 3. Relative Zentrierung auf die Hüft-Mitte
        for i in range(len(sequence)):
            # Wir arbeiten jetzt mit (12, 3) -> x, y, conf
            frame_data = sequence[i]

            # Hüft-Koordinaten nur von X und Y (Index 0 und 1)
            hip_l = frame_data[6, :2]
            hip_r = frame_data[7, :2]

            # Prüfen, ob Hüft-Punkte vorhanden sind (Confidence > 0 ist hier meist sicherer als != 0)
            if not (np.all(hip_l == 0) and np.all(hip_r == 0)):
                center = (hip_l + hip_r) / 2

                # Maske: Nur Gelenke zentrieren, die tatsächlich x,y Daten haben
                # Wir prüfen hier nur die ersten zwei Spalten
                mask = np.any(frame_data[:, :2] != 0, axis=1)

                # Der Fix für 3 Achsen:
                # Wir subtrahieren 'center' NUR von den ersten zwei Spalten der maskierten Gelenke
                frame_data[mask, :2] = frame_data[mask, :2] - center

                # Zurückschreiben in die sequence (Index 2/Confidence bleibt original)
                sequence[i] = frame_data

        return sequence

    def drop_confidence_axis(self):
        """
        Entfernt die Confidence-Achse (letzte Dimension), falls gewünscht.

        Erwartete Shapes:
        - (N, T, 12, 3) -> (N, T, 12, 2)
        - (T, 12, 3)    -> (T, 12, 2)
        """
        self.x = np.array(self.x)

        if self.x.shape[-1] != 3:
            raise ValueError(f"Expected last dimension == 3, got {self.x.shape}")

        self.x = self.x[..., :2]

    # for own created dataset old
    #
    #
    #
    #
    #

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
                    self.video_id.append(global_subject_counter)
                    self.is_original_data.append(True)

                    # augmentation
                    augmented_windows = self.augmenter.augment_window(new_sliding_window)
                    self.x.extend(augmented_windows)
                    for i in range(len(augmented_windows)):
                        self.y.append(gesture_id)
                        self.video_id.append(global_subject_counter)
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
        self.video_id = np.array(self.video_id)
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
                    self.video_id.append(global_subject_counter)
                    self.is_original_data.append(True)

                    # augmentation
                    augmented_windows = self.augmenter.augment_window(new_sliding_window)
                    self.x.extend(augmented_windows)
                    for i in range(len(augmented_windows)):
                        self.y.append(gesture_id)
                        self.video_id.append(global_subject_counter)
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
        self.video_id = np.array(self.video_id)
        self.is_original_data = np.array(self.is_original_data)

    def print_data_after_preparation(self):
        print("\n" + "=" * 30)
        print("DATA PREPARATION SUMMARY")
        print("=" * 30)
        print(f"Total Windows (Samples): {len(self.x)}")
        print(f"Window Shape (Frames x Features): {self.x.shape[1:] if len(self.x) > 0 else 'N/A'}")
        print(f"Total Unique Subjects: {len(np.unique(self.video_id))}")

        print("\nWindows per Gesture:")
        # Reverse lookup for gesture names
        id_to_label = {v: k for k, v in self.label_id_dic.items()}
        unique_labels, counts = np.unique(self.y, return_counts=True)

        for label_id, count in zip(unique_labels, counts):
            print(f" - {id_to_label[label_id]}: {count} windows")
        print("=" * 30 + "\n")
