from fastapi import FastAPI
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from typing import Sequence, Dict

app = FastAPI()
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
cap     = cv2.VideoCapture(0)

def calculate_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

JOINTS = {
    "right_knee":  (mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.RIGHT_KNEE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE),
    "left_knee":   (mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.LEFT_KNEE,
                    mp_pose.PoseLandmark.LEFT_ANKLE),
    "right_elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST),
    "left_elbow":  (mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST),
    "right_shoulder": (mp_pose.PoseLandmark.RIGHT_HIP,
                       mp_pose.PoseLandmark.RIGHT_SHOULDER,
                       mp_pose.PoseLandmark.RIGHT_ELBOW),
    "left_shoulder":  (mp_pose.PoseLandmark.LEFT_HIP,
                       mp_pose.PoseLandmark.LEFT_SHOULDER,
                       mp_pose.PoseLandmark.LEFT_ELBOW),
    "right_hip":   (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.RIGHT_KNEE),
    "left_hip":    (mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.LEFT_KNEE),
}

@app.get("/metrics")
def get_metrics() -> Dict[str, float]:
    ret, frame = cap.read()
    if not ret:
        return JSONResponse({"error": "no frame"})
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return {name: None for name in JOINTS}

    lm = results.pose_landmarks.landmark
    h, w = frame.shape[:2]
    angles: Dict[str, float] = {}

    for name, (A, B, C) in JOINTS.items():
        a = [lm[A].x * w, lm[A].y * h]
        b = [lm[B].x * w, lm[B].y * h]
        c = [lm[C].x * w, lm[C].y * h]
        angles[name] = calculate_angle(a, b, c)

    return angles
