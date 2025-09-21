import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANGLES_CSV = ROOT / "dataset" / "video_angles.csv"
WINDOWS_NPZ = ROOT / "dataset" / "windows_phase2.npz"
SUMMARY_CSV = ROOT / "dataset" / "windows_phase2_summary.csv"

SEQ_LEN = 200
STRIDE = 10

def norm_video_name(name: str) -> str:
    return str(name).replace("\u00A0", " ").strip().lower()

def infer_label_from_name(name: str) -> str:
    s = name.lower()
    if "jack" in s:
        return "Jumping Jacks"
    if "pull" in s and "up" in s:
        return "Pull ups"
    if "push" in s and "up" in s:
        return "Push Ups"
    if "twist" in s:
        return "Russian twists"
    if "squat" in s:
        return "Squats"
    return "Unknown"

def main():
    df = pd.read_csv(ANGLES_CSV, encoding="utf-8")
    df["video_norm"] = df["video_norm"].astype(str).apply(norm_video_name)
    df["video"] = df["video"].astype(str)
    df["label"] = df["video_norm"].apply(infer_label_from_name)
    numeric_cols = [c for c in df.columns if c not in ("video","video_norm","label","frame")]
    numeric_cols = [c for c in numeric_cols if df[c].dtype != object]
    X_list, y_list, meta = [], [], []

    for vnorm, d in df.groupby("video_norm"):
        d = d.sort_values("frame").reset_index(drop=True)
        arr = d[numeric_cols].to_numpy(dtype=np.float32)
        n = arr.shape[0]
        if n < SEQ_LEN:
            continue
        vid = d["video"].iloc[0]
        lab = d["label"].iloc[0]
        for start in range(0, n - SEQ_LEN + 1, STRIDE):
            seg = arr[start:start+SEQ_LEN]
            X_list.append(seg)
            y_list.append(lab)
            meta.append({"video": vid, "video_norm": vnorm, "start": int(start), "label": lab})
    if not X_list:
        return

    X = np.stack(X_list, axis=0).astype(np.float32)
    labels = sorted(set(y_list))
    label_to_idx = {k:i for i,k in enumerate(labels)}
    y = np.array([label_to_idx[s] for s in y_list], dtype=np.int64)
    mu = np.nanmean(X, axis=(0,1))
    sd = np.nanstd(X, axis=(0,1))
    sd = np.where(sd == 0, 1.0, sd)
    np.savez(WINDOWS_NPZ, X=X, y=y, mu=mu, sd=sd, seq_len=SEQ_LEN,feat_cols=np.array(numeric_cols), type_names=np.array(labels))
    summ = pd.DataFrame(meta)
    summ.insert(0, "window_idx", np.arange(len(summ), dtype=int))
    summ.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()

