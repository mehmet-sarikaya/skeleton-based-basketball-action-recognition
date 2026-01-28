import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SkeletonAnimator:
    """
    Animiert Keypoints über die Zeit (Skeleton), für COCO ohne Kopf (12 Joints).
    Unterstützte seq-Shapes:
      - (T, 12, 2)
      - (T, 12, 3)  (wird auf x,y reduziert)
      - (T, 12, 2, 1) (squeezed)
    Missing:
      - Frames komplett 0 -> wird als "missing frame" behandelt
      - einzelne Joints (0,0) -> werden nicht gezeichnet
    """

    COCO_NO_HEAD_NAMES = [
        "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow",
        "l_wrist", "r_wrist",
        "l_hip", "r_hip",
        "l_knee", "r_knee",
        "l_ankle", "r_ankle",
    ]

    # Bones für COCO ohne Kopf
    # Torso: l_shoulder - r_shoulder, l_hip - r_hip, l_shoulder - l_hip, r_shoulder - r_hip
    # Arme: shoulder - elbow - wrist
    # Beine: hip - knee - ankle
    COCO_NO_HEAD_BONES = [
        (0, 1), (6, 7), (0, 6), (1, 7),
        (0, 2), (2, 4),
        (1, 3), (3, 5),
        (6, 8), (8, 10),
        (7, 9), (9, 11),
    ]

    def __init__(
        self,
        bones=None,
        joint_names=None,
        invert_y=True,
        fps=10,
        show_joints=True,
        show_bones=True,
        show_joint_indices=False,
        pad_frame_allzero=True,
        pad_joint_allzero=True,
    ):
        self.bones = bones if bones is not None else self.COCO_NO_HEAD_BONES
        self.joint_names = joint_names if joint_names is not None else self.COCO_NO_HEAD_NAMES

        self.invert_y = invert_y
        self.interval_ms = int(1/fps * 1000)
        self.show_joints = show_joints
        self.show_bones = show_bones
        self.show_joint_indices = show_joint_indices

        self.pad_frame_allzero = pad_frame_allzero
        self.pad_joint_allzero = pad_joint_allzero

        # intern
        self._fig = None
        self._ax = None
        self._anim = None
        self._scat = None
        self._lines = []
        self._texts = []
        self._seq = None
        self._frame_valid = None

    @staticmethod
    def _to_TJ2(seq):
        seq = np.asarray(seq)
        if seq.ndim == 4 and seq.shape[-1] == 1:
            seq = np.squeeze(seq, axis=-1)
        if seq.ndim != 3:
            raise ValueError(f"Expected seq ndim=3 after squeeze, got shape={seq.shape}")
        if seq.shape[-1] < 2:
            raise ValueError(f"Expected last dim >= 2 (x,y), got shape={seq.shape}")
        if seq.shape[-1] > 2:
            seq = seq[..., :2]
        return seq.astype(np.float32)

    def _compute_bounds(self, seq):
        # gültige Punkte (nicht 0,0)
        pts = seq.reshape(-1, 2)
        mask = np.any(pts != 0.0, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            # fallback
            return (-1, 1, -1, 1)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        # padding
        dx = xmax - xmin
        dy = ymax - ymin
        pad_x = 0.1 * (dx if dx > 1e-6 else 1.0)
        pad_y = 0.1 * (dy if dy > 1e-6 else 1.0)
        return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    def animate(self, seq, title=None):
        """
        Startet die Animation und gibt das FuncAnimation-Objekt zurück.
        """
        seq = self._to_TJ2(seq)
        T, J, _ = seq.shape

        if J != 12:
            raise ValueError(f"Expected 12 joints for COCO-no-head, got J={J}")

        self._seq = seq

        # frame-valid: mindestens ein Joint != 0
        if self.pad_frame_allzero:
            self._frame_valid = np.any(seq != 0.0, axis=(1, 2))
        else:
            self._frame_valid = np.ones((T,), dtype=bool)

        xmin, xmax, ymin, ymax = self._compute_bounds(seq)

        self._fig, self._ax = plt.subplots()
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlim(xmin, xmax)
        self._ax.set_ylim(ymin, ymax)
        self._ax.grid(True, alpha=0.3)
        if self.invert_y:
            self._ax.invert_yaxis()
        if title:
            self._ax.set_title(title)

        # artists
        if self.show_joints:
            self._scat = self._ax.scatter([], [])
        else:
            self._scat = None

        self._lines = []
        if self.show_bones:
            for _ in self.bones:
                (ln,) = self._ax.plot([], [])
                self._lines.append(ln)

        self._texts = []
        if self.show_joint_indices:
            for j in range(J):
                txt = self._ax.text(0, 0, "", fontsize=8)
                self._texts.append(txt)

        def init():
            if self._scat is not None:
                self._scat.set_offsets(np.zeros((0, 2)))
            for ln in self._lines:
                ln.set_data([], [])
            for txt in self._texts:
                txt.set_text("")
            return [a for a in ([self._scat] if self._scat is not None else [])] + self._lines + self._texts

        def update(t):
            frame = self._seq[t]  # (12,2)

            if not self._frame_valid[t]:
                # missing frame
                if self._scat is not None:
                    self._scat.set_offsets(np.zeros((0, 2)))
                for ln in self._lines:
                    ln.set_data([], [])
                for txt in self._texts:
                    txt.set_text("")
                self._ax.set_xlabel(f"Frame {t}/{T-1} (MISSING/PAD)")
                return [a for a in ([self._scat] if self._scat is not None else [])] + self._lines + self._texts

            # joint-valid
            if self.pad_joint_allzero:
                joint_valid = np.any(frame != 0.0, axis=1)
            else:
                joint_valid = np.ones((J,), dtype=bool)

            if self._scat is not None:
                self._scat.set_offsets(frame[joint_valid])

            if self.show_bones:
                for k, (i, j) in enumerate(self.bones):
                    if joint_valid[i] and joint_valid[j]:
                        self._lines[k].set_data([frame[i, 0], frame[j, 0]],
                                                [frame[i, 1], frame[j, 1]])
                    else:
                        self._lines[k].set_data([], [])

            if self.show_joint_indices:
                for idx in range(J):
                    if joint_valid[idx]:
                        self._texts[idx].set_position((frame[idx, 0], frame[idx, 1]))
                        self._texts[idx].set_text(str(idx))
                    else:
                        self._texts[idx].set_text("")

            self._ax.set_xlabel(f"Frame {t}/{T-1}")
            return [a for a in ([self._scat] if self._scat is not None else [])] + self._lines + self._texts

        self._anim = FuncAnimation(
            self._fig,
            update,
            frames=T,
            init_func=init,
            interval=self.interval_ms,
            blit=False
        )
        plt.show()
        return self._anim

    def save(self, filepath, fps=None, dpi=120):
        """
        Speichert die zuletzt erstellte Animation als GIF/MP4.
        """
        if self._anim is None:
            raise RuntimeError("No animation to save. Call animate(...) first.")
        if fps is None:
            fps = max(1, int(1000 / self.interval_ms))

        ext = filepath.lower().split(".")[-1]

        if ext == "mp4":
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            self._anim.save(filepath, writer=writer, dpi=dpi)

        elif ext == "gif":
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            self._anim.save(filepath, writer=writer, dpi=dpi)

        else:
            raise ValueError(f"Unsupported file extension: .{ext}")
