import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANGLES_CSV = ROOT / "dataset" / "video_angles.csv"
OUT_CSV = ROOT / "dataset" / "video_labels.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

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
    vids = df[["video_norm"]].drop_duplicates().reset_index(drop=True)
    vids["label"] = vids["video_norm"].apply(infer_label_from_name)
    vids.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
