import os
import sys
import json
import cv2
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "yolov/yolo26x-pose.pt"
EXAMPLES_DIR = Path("space_jam/examples")
OUT_DIR = Path("annotated_video")

LABEL_ID_DIC = {
    0: "block",
    1: "pass",
    2: "run",
    3: "dribble",
    4: "shoot",
    5: "ball in hand",
    6: "defense",
    7: "pick",
    8: "no_action",
    9: "walk",
}

ALLOWED_EXTS = [
    ".mp4", ".mkv", ".mov", ".avi",
    ".wmv", ".webm", ".flv", ".m4v"
]

# ============================================================
# LOAD ANNOTATION DICT
# ============================================================
def load_annotation_dict(path: Path):
    dict_path = path / "annotation_dict.json"

    if not dict_path.exists():
        dict_path = path.parent / "annotation_dict.json"

    if not dict_path.exists():
        raise FileNotFoundError(
            f"annotation_dict.json wurde weder in {path} noch in {path.parent} gefunden!"
        )

    with dict_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # keys: video_id (string), values: label_id (int)
    return {str(k): int(v) for k, v in data.items()}

# ============================================================
# HELPERS
# ============================================================
def find_video(stem: str) -> Path | None:
    for ext in ALLOWED_EXTS:
        p = EXAMPLES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def safe_label_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def export_video(
    video_path: Path,
    out_path: Path,
    model: YOLO,
    annotate: bool
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV konnte Video nicht öffnen: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    if w == 0 or h == 0:
        cap.release()
        raise ValueError(f"Ungültige Videodimensionen: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    if not out.isOpened():
        cap.release()
        raise IOError(f"VideoWriter konnte nicht öffnen: {out_path}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if annotate:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False
                )

                if results and len(results) > 0:
                    out.write(results[0].plot())
                else:
                    out.write(frame)
            else:
                # RAW export
                out.write(frame)

    finally:
        cap.release()
        out.release()

# ============================================================
# MAIN
# ============================================================
def main():
    print("→ Lade Annotation Dict")
    annotation_dict = load_annotation_dict(EXAMPLES_DIR)

    print("→ Lade YOLO Modell")
    model = YOLO(MODEL_PATH)

    # genau ein Video pro Label wählen
    chosen = {}  # label_id -> Path

    for video_id, label_id in annotation_dict.items():
        if label_id not in LABEL_ID_DIC:
            continue
        if label_id in chosen:
            continue

        vp = find_video(video_id)
        if vp is None:
            continue

        chosen[label_id] = vp

        if len(chosen) == len(LABEL_ID_DIC):
            break

    missing = [lid for lid in LABEL_ID_DIC if lid not in chosen]
    if missing:
        print("⚠️  Keine Videos gefunden für Labels:")
        for lid in missing:
            print(f"  {lid}: {LABEL_ID_DIC[lid]}")

    if not chosen:
        print("❌ Keine passenden Videos gefunden.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label_id, video_path in chosen.items():
        label_name = safe_label_name(LABEL_ID_DIC[label_id])
        ext = video_path.suffix

        raw_out = OUT_DIR / f"{label_name}_raw{ext}"
        ann_out = OUT_DIR / f"{label_name}_annotated{ext}"

        print(f"\n▶ Label: {LABEL_ID_DIC[label_id]}")
        print(f"  Input     : {video_path}")
        print(f"  Raw       : {raw_out}")
        print(f"  Annotated : {ann_out}")

        export_video(video_path, raw_out, model, annotate=False)
        export_video(video_path, ann_out, model, annotate=True)

        print("  ✓ fertig")

    print("\n✔ Alle Videos verarbeitet.")

# ============================================================
if __name__ == "__main__":
    main()