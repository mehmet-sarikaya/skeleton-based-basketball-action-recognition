import cv2 as cv
import math
import os


class VideoProcessor:
    def __init__(self):
        pass

    def ProcessVideoPerFrame(self, video_path: str, frame_callback, target_fps=10, show_frames=False):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

        cap = cv.VideoCapture(video_path)
        original_fps = math.ceil(cap.get(cv.CAP_PROP_FPS))

        if target_fps > original_fps:
            raise ValueError(
                f"Konfigurationsfehler: target_fps ist zu hoch!\n"
                f"Eingestellt: {target_fps} FPS\n"
                f"Video-Limit: {original_fps} FPS"
            )

        print("original fps:", original_fps)
        print("target fps:", target_fps)

        num_frames_to_skip = math.ceil(original_fps / target_fps)

        print("to achieve target fps - frames to skip:", num_frames_to_skip)

        idx = - 1

        while cap.isOpened():
            successful, frame = cap.read()
            if successful:
                idx += 1
                if idx % 3 != 0:
                    continue

                if show_frames:
                    cv.imshow("First Frame", frame)
                    cv.waitKey()

                # process each frame with the given callback
                frame_callback(frame)

            else:
                print("Couldn't read frame")
                # release video capture
                cap.release()
                cv.destroyAllWindows()