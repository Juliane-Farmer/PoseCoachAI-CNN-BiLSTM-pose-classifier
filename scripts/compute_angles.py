import os
import numpy as np
import pandas as pd

IN_CSV  = os.path.join('dataset', 'video_keypoints.csv')
OUT_CSV = os.path.join('dataset', 'video_angles.csv')
VIS_THRESH = 0.5

def compute_angle(a, b, c):
    """
    Angle at point b formed by segments (b->a) and (b->c), in degrees.
    Returns NaN if any vector is degenerate.
    """
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def get_xyzv(row, name):
    """Return (xyz, vis) for landmark NAME_* columns, or (None, vis) if missing."""
    try:
        x = row[f'{name}_x']; y = row[f'{name}_y']; z = row[f'{name}_z']; v = row[f'{name}_v']
        return np.array([x, y, z], dtype=np.float32), float(v)
    except KeyError:
        return None, 0.0

def safe_joint_angle(row, A, B, C):
    """
    Compute angle A–B–C if all three landmarks exist and have sufficient visibility.
    """
    a, va = get_xyzv(row, A)
    b, vb = get_xyzv(row, B)
    c, vc = get_xyzv(row, C)
    if a is None or b is None or c is None:
        return np.nan
    if (va < VIS_THRESH) or (vb < VIS_THRESH) or (vc < VIS_THRESH):
        return np.nan
    return compute_angle(a, b, c)

def trunk_tilt_deg(row):
    """
    Angle between torso (mid-hip -> mid-shoulder) and the image vertical (0, -1, 0).
    0° ≈ upright; larger = leaning. Uses absolute angle.
    """
    ls, vs = get_xyzv(row, 'LEFT_SHOULDER')
    rs, vsr = get_xyzv(row, 'RIGHT_SHOULDER')
    lh, vh = get_xyzv(row, 'LEFT_HIP')
    rh, vhr = get_xyzv(row, 'RIGHT_HIP')
    if any(v is None for v in [ls, rs, lh, rh]):
        return np.nan
    if min(vs, vsr, vh, vhr) < VIS_THRESH:
        return np.nan
    mid_sh = (ls + rs) / 2.0
    mid_hip = (lh + rh) / 2.0
    torso = mid_sh - mid_hip
    n = np.linalg.norm(torso)
    if n == 0:
        return np.nan
    torso = torso / n
    vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # up in image coords
    cosang = float(np.clip(np.dot(torso, vertical), -1.0, 1.0))
    return np.degrees(np.arccos(cosang))

df = pd.read_csv(IN_CSV)

angles = {
    'right_elbow_angle': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    'left_elbow_angle' : ('LEFT_SHOULDER',  'LEFT_ELBOW',  'LEFT_WRIST'),

    'right_knee_angle' : ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    'left_knee_angle'  : ('LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE'),

    'right_hip_angle'  : ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
    'left_hip_angle'   : ('LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'),

    'right_shoulder_abd': ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    'left_shoulder_abd' : ('LEFT_HIP',  'LEFT_SHOULDER',  'LEFT_ELBOW'),}

for out_col, (A, B, C) in angles.items():
    df[out_col] = [safe_joint_angle(row, A, B, C) for _, row in df.iterrows()]

df['trunk_tilt'] = [trunk_tilt_deg(row) for _, row in df.iterrows()]

angle_cols = list(angles.keys()) + ['trunk_tilt']

def add_temporal_features(g):
    for c in angle_cols:
        g[f'{c}_diff'] = g[c].diff()
        g[f'{c}_ma5']  = g[c].rolling(window=5, min_periods=1).mean()
    g[angle_cols] = g[angle_cols].interpolate(limit_direction='both')
    return g

if 'video' in df.columns:
    df = df.sort_values(['video', 'frame']).groupby('video', group_keys=False).apply(add_temporal_features)
else:
    df = df.sort_values('frame')
    df = add_temporal_features(df)

df.to_csv(OUT_CSV, index=False)

print(f"Wrote joint angles (+ temporal features) to {OUT_CSV}")
print(f"Angle features added: {', '.join(angle_cols)}")
print("Temporal features: *_diff, *_ma5")


