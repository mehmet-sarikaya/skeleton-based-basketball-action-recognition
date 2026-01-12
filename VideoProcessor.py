import cv2 as cv
import math
import os


class VideoProcessor:
    def __init__(self, target_fps):
        self.target_fps = target_fps

    def process_video_per_frame_with_fixed_frame_skipping(self, video_path: str, frame_callback, show_frames=False):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

        cap = cv.VideoCapture(video_path)
        original_fps = math.ceil(cap.get(cv.CAP_PROP_FPS))

        if self.target_fps > original_fps:
            raise ValueError(
                f"Konfigurationsfehler: target_fps ist zu hoch!\n"
                f"Eingestellt: {self.target_fps} FPS\n"
                f"Video-Limit: {original_fps} FPS"
            )

        print("original fps:", original_fps)
        print("target fps:", self.target_fps)

        num_frames_to_skip = math.ceil(original_fps / self.target_fps)

        print("to achieve target fps - frame steps:", num_frames_to_skip)

        idx = - 1

        while cap.isOpened():
            successful, frame = cap.read()
            if successful:
                idx += 1
                if idx % num_frames_to_skip != 0:
                    continue

                if show_frames:
                    cv.imshow("First Frame", frame)
                    cv.waitKey()

                # process each frame with the given callback
                frame_callback(frame)

            else:
                print("End of Video")
                # release video capture
                cap.release()
                cv.destroyAllWindows()

    def process_video_per_frame_at_constant_fps(self, video_path: str, frame_callback, show_frames=False):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

        cap = cv.VideoCapture(video_path)
        original_fps = math.ceil(cap.get(cv.CAP_PROP_FPS))

        if original_fps <= 0:
            print("Warnung: FPS konnte nicht gelesen werden.")
            return

        if self.target_fps > original_fps:
            raise ValueError(
                f"Konfigurationsfehler: target_fps ist zu hoch!\n"
                f"Eingestellt: {self.target_fps} FPS\n"
                f"Video-Limit: {original_fps} FPS"
            )

        print(f"Original FPS: {original_fps:.2f} -> Target FPS: {self.target_fps}")

        target_interval = 1.0 / self.target_fps
        current_video_time = 0.0
        next_process_time = 0.0

        idx = 0
        while cap.isOpened():
            successful, frame = cap.read()
            if not successful:
                break

            current_video_time = idx / original_fps

            if current_video_time >= next_process_time:
                if show_frames:
                    cv.imshow("Processing Frame", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_callback(frame)

                next_process_time += target_interval

            idx += 1

        cap.release()
        cv.destroyAllWindows()
        print("End of Video")
