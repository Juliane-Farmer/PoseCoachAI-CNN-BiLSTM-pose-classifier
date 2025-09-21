import os
from pathlib import Path
import math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KP_DIR = ROOT / "dataset" / "keypoints"
OUT_CSV = ROOT / "dataset" / "video_angles.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

VIS_THRESH = 0.5

IDX = {
    "NOSE": 0,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,}

def norm_video_name(name: str) -> str:
    return str(name).replace("\u00A0", " ").strip().lower()

def angle_at(b, a, c):
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def trunk_tilt_deg(shoulder, hip):
    v = shoulder - hip
    if np.linalg.norm(v) == 0:
        return np.nan
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cosang = np.clip(np.dot(v / np.linalg.norm(v), up), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return float(ang)

def has_vis(row, i):
    return float(row.get(f"v{i}", 0.0)) >= VIS_THRESH

def xyz(row, i):
    return np.array([float(row.get(f"x{i}", 0.0)), float(row.get(f"y{i}", 0.0)), float(row.get(f"z{i}", 0.0))], dtype=np.float32)

def compute_features(df):
    out = []
    for _, r in df.iterrows():
        feats = {"frame": int(r["frame"])}
        if all(has_vis(r, IDX[k]) for k in ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"]):
            feats["right_elbow_angle"] = angle_at(xyz(r, IDX["RIGHT_ELBOW"]), xyz(r, IDX["RIGHT_SHOULDER"]), xyz(r, IDX["RIGHT_WRIST"]))
        else:
            feats["right_elbow_angle"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"]):
            feats["left_elbow_angle"] = angle_at(xyz(r, IDX["LEFT_ELBOW"]), xyz(r, IDX["LEFT_SHOULDER"]), xyz(r, IDX["LEFT_WRIST"]))
        else:
            feats["left_elbow_angle"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"]):
            feats["right_knee_angle"] = angle_at(xyz(r, IDX["RIGHT_KNEE"]), xyz(r, IDX["RIGHT_HIP"]), xyz(r, IDX["RIGHT_ANKLE"]))
        else:
            feats["right_knee_angle"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"]):
            feats["left_knee_angle"] = angle_at(xyz(r, IDX["LEFT_KNEE"]), xyz(r, IDX["LEFT_HIP"]), xyz(r, IDX["LEFT_ANKLE"]))
        else:
            feats["left_knee_angle"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["RIGHT_SHOULDER","RIGHT_HIP","RIGHT_KNEE"]):
            feats["right_hip_angle"] = angle_at(xyz(r, IDX["RIGHT_HIP"]), xyz(r, IDX["RIGHT_SHOULDER"]), xyz(r, IDX["RIGHT_KNEE"]))
        else:
            feats["right_hip_angle"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["LEFT_SHOULDER","LEFT_HIP","LEFT_KNEE"]):
            feats["left_hip_angle"] = angle_at(xyz(r, IDX["LEFT_HIP"]), xyz(r, IDX["LEFT_SHOULDER"]), xyz(r, IDX["LEFT_KNEE"]))
        else:
            feats["left_hip_angle"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["LEFT_SHOULDER","LEFT_HIP"]):
            feats["trunk_tilt"] = trunk_tilt_deg(xyz(r, IDX["LEFT_SHOULDER"]), xyz(r, IDX["LEFT_HIP"]))
        elif all(has_vis(r, IDX[k]) for k in ["RIGHT_SHOULDER","RIGHT_HIP"]):
            feats["trunk_tilt"] = trunk_tilt_deg(xyz(r, IDX["RIGHT_SHOULDER"]), xyz(r, IDX["RIGHT_HIP"]))
        else:
            feats["trunk_tilt"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["LEFT_ELBOW","LEFT_SHOULDER","LEFT_HIP"]):
            feats["left_shoulder_abd"] = angle_at(xyz(r, IDX["LEFT_SHOULDER"]), xyz(r, IDX["LEFT_ELBOW"]), xyz(r, IDX["LEFT_HIP"]))
        else:
            feats["left_shoulder_abd"] = np.nan
        if all(has_vis(r, IDX[k]) for k in ["RIGHT_ELBOW","RIGHT_SHOULDER","RIGHT_HIP"]):
            feats["right_shoulder_abd"] = angle_at(xyz(r, IDX["RIGHT_SHOULDER"]), xyz(r, IDX["RIGHT_ELBOW"]), xyz(r, IDX["RIGHT_HIP"]))
        else:
            feats["right_shoulder_abd"] = np.nan
        out.append(feats)
    return pd.DataFrame(out)

def add_temporal_features(df):
    df = df.sort_values("frame").reset_index(drop=True)
    base_cols = [c for c in df.columns if c not in ("frame","video","video_norm")]
    for c in base_cols:
        df[f"{c}_diff"] = df[c].diff()
        df[f"{c}_ma5"] = df[c].rolling(5, min_periods=1).mean()
    return df

def load_keypoints_csv(p):
    df = pd.read_csv(p, encoding="utf-8")
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df), dtype=int)
    return df

def main():
    rows = []
    for csvf in sorted(KP_DIR.glob("*.csv")):
        stem = csvf.stem
        vid = stem + ".mp4"
        df = load_keypoints_csv(csvf)
        feat = compute_features(df)
        feat["video"] = vid
        feat["video_norm"] = norm_video_name(vid)
        rows.append(feat)
    if not rows:
        return
    angles = pd.concat(rows, axis=0, ignore_index=True)
    angles = angles.groupby("video_norm", group_keys=False).apply(add_temporal_features)
    angles.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()


