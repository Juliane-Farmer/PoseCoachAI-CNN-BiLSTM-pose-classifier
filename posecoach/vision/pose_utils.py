from typing import Tuple
import numpy as np
import mediapipe as mp

VIS_THRESH = 0.5
POSE_LM = mp.solutions.pose.PoseLandmark

ANGLE_SPECS = {
    "right_elbow_angle": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_elbow_angle":  ("LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST"),
    "right_knee_angle":  ("RIGHT_HIP",      "RIGHT_KNEE",  "RIGHT_ANKLE"),
    "left_knee_angle":   ("LEFT_HIP",       "LEFT_KNEE",   "LEFT_ANKLE"),
    "right_hip_angle":   ("RIGHT_SHOULDER", "RIGHT_HIP",   "RIGHT_KNEE"),
    "left_hip_angle":    ("LEFT_SHOULDER",  "LEFT_HIP",    "LEFT_KNEE"),
    "right_shoulder_abd": ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    "left_shoulder_abd":  ("LEFT_HIP",  "LEFT_SHOULDER",  "LEFT_ELBOW"),}

REQUIRED_JOINTS = {
    "squat": ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"],
    "squats": ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"],
    "russian twists": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"],
    "russian twist": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"],
    "jumping jacks": ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_SHOULDER","RIGHT_SHOULDER"],
    "pull ups": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_HIP","RIGHT_HIP"],}

def get_xyzv(lm, name: str) -> Tuple[np.ndarray, float]:
    idx = getattr(POSE_LM, name); pt = lm[idx]
    return np.array([pt.x, pt.y, pt.z], dtype=np.float32), float(pt.visibility)

def angle(a, b, c) -> float:
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0: return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def safe_joint_angle(lm, A, B, C) -> float:
    a, va = get_xyzv(lm, A); b, vb = get_xyzv(lm, B); c, vc = get_xyzv(lm, C)
    if min(va, vb, vc) < VIS_THRESH: return np.nan
    return angle(a, b, c)

def trunk_tilt_deg(lm) -> float:
    ls, vs = get_xyzv(lm, "LEFT_SHOULDER"); rs, vsr = get_xyzv(lm, "RIGHT_SHOULDER")
    lh, vh = get_xyzv(lm, "LEFT_HIP"); rh, vhr = get_xyzv(lm, "RIGHT_HIP")
    if min(vs, vsr, vh, vhr) < VIS_THRESH: return np.nan
    mid_sh = (ls + rs) / 2.0; mid_hip = (lh + rh) / 2.0
    torso = mid_sh - mid_hip; n = np.linalg.norm(torso)
    if n == 0: return np.nan
    torso = torso / n; vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cosang = float(np.clip(np.dot(torso, vertical), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))
