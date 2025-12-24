import cv2
from ultralytics import YOLO
import torch

# Load the YOLO11 model
model = YOLO("yolov/yolo11n-pose.pt")

img = cv2.imread("../Images/Wurf/kobe.png")

results = model(img)

print("Anzahl Frames glaub?! :", len(results))

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)

    for idx, e in enumerate(xyn):
        print(f"{idx}.Eintrag", e)
"""
result.keypoints.xyn ist ein Tensor der Form (N, K, 2):

N = Anzahl erkannter Personen im Bild (bei dir offenbar 1 → daher nur 0.Eintrag)

K = Anzahl Keypoints (17 beim COCO-Standardmodell yolo11n-pose.pt)

2 = [x,y]

Die Werte sind normalisiert auf Bildbreite/-höhe:

x=0.4754
x=0.4754 bedeutet: 47.54% der Bildbreite von links

y=0.3025
y=0.3025 bedeutet: 30.25% der Bildhöhe von oben 
docs.ultralytics.com
+1

Dass da device='cuda:0' steht, heißt: Der Tensor liegt auf der GPU.
"""

r = results[0]
h, w = r.orig_shape  # (height, width)

kpts = r.keypoints.data[0].cpu().numpy()  # Person 0, (17,3)

KPT_NAMES = [
  "nose","left_eye","right_eye","left_ear","right_ear",
  "left_shoulder","right_shoulder","left_elbow","right_elbow",
  "left_wrist","right_wrist","left_hip","right_hip",
  "left_knee","right_knee","left_ankle","right_ankle"
]
# Augen, Ohren, und Nase unwichtig
# 0,1,2,3,4 unwichtig!

for name, (x, y, conf) in zip(KPT_NAMES, kpts):
    xn, yn = x / w, y / h
    print(f"{name:14s} px=({x:7.1f},{y:7.1f})  norm=({xn:.4f},{yn:.4f})  conf={conf:.3f}")


######################### imgshow
annotated_frame = results[0].plot()

cv2.imshow("YOLO11 Tracking", annotated_frame)
cv2.waitKey(0)                # WICHTIG: sonst schließt das Fenster sofort
cv2.destroyAllWindows()