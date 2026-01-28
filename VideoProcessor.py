import cv2 as cv
import math
import os
import time
import cv2
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

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

    def process_video_per_frame_at_constant_fps(self, video_path: str, frame_callback, show_frames=False, verbose=True):

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
        if verbose:
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
        if verbose:
            print("End of Video")

    def process_camera_per_frame_at_constant_fps(
            self,
            frame_callback,
            show_frames=False
    ):
        # available_ports, working_ports = self.list_ports()
        # camera_id = working_ports[0]
        # camera_id = 0
        camera_id = "http://134.103.92.227:8080/video"
        cap = cv.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        print(f"Camera {camera_id} opened. Target FPS: {self.target_fps}")

        target_interval = 1.0 / self.target_fps
        next_process_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            if now >= next_process_time:
                if show_frames:
                    cv.imshow("Camera", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_callback(frame)
                next_process_time += target_interval

            # Optional: prevent busy looping
            time.sleep(0.001)

        cap.release()
        cv.destroyAllWindows()
        print("Camera stopped")

    def list_ports(self):
        """
        Test the ports and returns a tuple with the available ports
        and the ones that are working.
        """
        is_working = True
        dev_port = 0
        working_ports = []
        available_ports = []
        while is_working:
            camera = cv.VideoCapture(dev_port)
            if not camera.isOpened():
                is_working = False
                print("Port %s is not working." % dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                    working_ports.append(dev_port)
                else:
                    print("Port %s for camera ( %s x %s) is present but does not reads." % (dev_port, h, w))
                    available_ports.append(dev_port)
            dev_port += 1
        return available_ports, working_ports


def analyze_video_lengths(path):
    path = Path(path)
    bins = defaultdict(int)

    video_files = list(path.glob("*.mp4"))

    for video_path in tqdm(video_files, desc="Analyzing videos"):
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        if fps <= 0:
            continue

        duration = frames / fps
        bucket = int(duration * 10)  # 0.1s buckets

        bins[bucket] += 1

    print("\nStatistik (0.1s Buckets):")
    for k in sorted(bins):
        start = k / 10
        end = (k + 1) / 10
        print(f"{start:.1f} - {end:.1f} Sekunden: {bins[k]} Videos")

if __name__ == "__main__":
    analyze_video_lengths("space_jam/examples")
