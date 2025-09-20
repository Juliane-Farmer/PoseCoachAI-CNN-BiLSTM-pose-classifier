import os
import csv
import cv2
import mediapipe as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "videos"
OUT_DIR = ROOT / "dataset" / "keypoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_pose = mp.solutions.pose

def extract_keypoints_from_video(video_path, csv_path):
    cap = cv2.VideoCapture(str(video_path))
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = None
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            row = {"frame": frame_idx}
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    row[f"x{i}"] = lm.x
                    row[f"y{i}"] = lm.y
                    row[f"z{i}"] = lm.z
                    row[f"v{i}"] = lm.visibility
            else:
                for i in range(33):
                    row[f"x{i}"] = 0.0
                    row[f"y{i}"] = 0.0
                    row[f"z{i}"] = 0.0
                    row[f"v{i}"] = 0.0
            if writer is None:
                header = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
            writer.writerow(row)
            frame_idx += 1
    cap.release()

def main():
    for video_file in DATA_DIR.glob("*.mp4"):
        out_file = OUT_DIR / (video_file.stem + ".csv")
        if out_file.exists():
            print(f"Skipping {video_file.name}, already processed")
            continue
        print(f"Processing {video_file.name} → {out_file.name}")
        extract_keypoints_from_video(video_file, out_file)

if __name__ == "__main__":
    main()


