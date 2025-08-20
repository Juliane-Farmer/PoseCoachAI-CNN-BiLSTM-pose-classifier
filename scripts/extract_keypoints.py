import os
import cv2
import csv
import numpy as np
import mediapipe as mp

VIDEO_DIR = 'dataset/videos/'    
OUTPUT_CSV = 'dataset/video_keypoints.csv'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
landmark_names = [f"{lm.name}_{axis}" for lm in mp_pose.PoseLandmark for axis in ('x','y','z','v')]
header = ['video','frame'] + landmark_names

with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for video_name in video_files:
        video_path = os.path.join(VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                row = [video_name, frame_idx]
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                writer.writerow(row)
            frame_idx += 1
        cap.release()

pose.close()
print(f"Finished extracting keypoints to {OUTPUT_CSV}")
