import os
import numpy as np
import pandas as pd

def compute_angle(a, b, c):
    """Compute angle at point b formed by points a–b–c (in degrees)."""
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

IN_CSV  = os.path.join('dataset', 'video_keypoints.csv')
OUT_CSV = os.path.join('dataset', 'video_angles.csv')
df = pd.read_csv(IN_CSV)

A_shoulder = 'RIGHT_SHOULDER'
A_elbow    = 'RIGHT_ELBOW'
A_wrist    = 'RIGHT_WRIST'

B_hip      = 'RIGHT_HIP'
B_knee     = 'RIGHT_KNEE'
B_ankle    = 'RIGHT_ANKLE'

elbow_angles = []
knee_angles  = []

for _, row in df.iterrows():
    shoulder = np.array([row[f'{A_shoulder}_x'],
                         row[f'{A_shoulder}_y'],
                         row[f'{A_shoulder}_z']])
    elbow    = np.array([row[f'{A_elbow}_x'],
                         row[f'{A_elbow}_y'],
                         row[f'{A_elbow}_z']])
    wrist    = np.array([row[f'{A_wrist}_x'],
                         row[f'{A_wrist}_y'],
                         row[f'{A_wrist}_z']])
    hip      = np.array([row[f'{B_hip}_x'],
                         row[f'{B_hip}_y'],
                         row[f'{B_hip}_z']])
    knee     = np.array([row[f'{B_knee}_x'],
                         row[f'{B_knee}_y'],
                         row[f'{B_knee}_z']])
    ankle    = np.array([row[f'{B_ankle}_x'],
                         row[f'{B_ankle}_y'],
                         row[f'{B_ankle}_z']])
    elbow_angles.append(compute_angle(shoulder, elbow, wrist))
    knee_angles.append(compute_angle(hip, knee, ankle))

df['right_elbow_angle'] = elbow_angles
df['right_knee_angle']  = knee_angles

df.to_csv(OUT_CSV, index=False)
print(f"Wrote joint angles to {OUT_CSV}")
