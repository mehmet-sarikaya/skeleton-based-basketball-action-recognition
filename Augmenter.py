import numpy as np
from scipy.interpolate import interp1d


class Augmenter:
    def __init__(self, fps):
        self.fps = fps

    def augment_window(self, window):
        augmented_windows = []

        augmented_windows.append(self.switch_left_and_right_window(window))
        augmented_windows.append(self.stretch_squeeze_keypoints(window))

        return augmented_windows

    def switch_left_and_right_window(self, window):
        augmented_window = []
        for frame in window:
            augmented_frame = self.switch_left_and_right_one_frame(frame)
            augmented_window.append(augmented_frame)

        return augmented_window

    def stretch_squeeze_keypoints(self, data_window):
        # ("one window", data_window)
        num_frames = len(data_window)
        duration_seconds = num_frames / self.fps  # e.g., 20 frames at 20 fps = 1.0 second

        # Create the time axis based on actual time
        t_orig = np.linspace(0, duration_seconds, num_frames)

        # speed_factor < 1.0 -> Faster (Squeeze)
        # speed_factor > 1.0 -> Slower (Stretch)
        speed_factor = np.random.uniform(0.85, 1.15)

        new_duration = duration_seconds * speed_factor

        new_num_frames = int(new_duration * self.fps)

        ##################
        # We map the new number of samples across the original tim+e span
        t_new = np.linspace(0, duration_seconds, new_num_frames)

        num_keypoints = len(data_window[0])
        augmented_all_keypoints_seq = np.zeros((new_num_frames, num_keypoints, 2))

        for i in range(num_keypoints):
            one_keypoint_sequence = data_window[:, i, :]
            # print("one_keypoint_sequence", one_keypoint_sequence)

            augmented_all_keypoints_seq[:, i, :] = (
                self.stretch_squeeze_one_keypoint_per_window(one_keypoint_sequence, t_orig, t_new)
            )

        # print("Augmented Window Stretched")

        return augmented_all_keypoints_seq

    def stretch_squeeze_one_keypoint_per_window(self, data, t_orig, t_new):

        x = data[:, 0]
        y = data[:, 1]

        # print(x)
        # print(y)

        # 'cubic' ensures the curve remains smooth despite changing the point density
        f_x = interp1d(t_orig, x, kind='cubic')
        f_y = interp1d(t_orig, y, kind='cubic')

        # Generate the final coordinates
        augmented_data = np.column_stack((f_x(t_new), f_y(t_new)))

        return augmented_data

    def switch_left_and_right_one_frame(self, data):
        augmented = []
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                augmented.append(data[i + 1])
                augmented.append(data[i])
            else:
                augmented.append(data[i])

        augmented_np = np.array(augmented)

        # for mirroring
        augmented_np[:, 0] = 1.0 - augmented_np[:, 0]

        return augmented_np
