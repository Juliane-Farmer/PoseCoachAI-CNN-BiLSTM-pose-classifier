import os
import cv2
import pandas as pd

VID_DIR = "dataset/videos"
OUT_CSV = "dataset/video_labels.csv"

def infer_exercise(name: str):
    n = name.lower()
    if "jump" in n:
        return "Jumping Jacks"
    if "pull" in n:
        return "Pull ups"
    if "push" in n:
        return "Push Ups"
    if "squat" in n:
        return "Squats"
    if "russian" in n or "twist" in n:
        return "Russian twists"
    return None

def get_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 999999 
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total if total > 0 else 999999

rows = []
for f in os.listdir(VID_DIR):
    if not f.lower().endswith((".mp4", ".avi", ".mov")):
        continue
    if "&" in f:
        print(f"[skip] mixed exercise file: {f}")
        continue
    label = infer_exercise(f)
    if not label:
        print(f"[warn] unknown exercise for {f}, skipping.")
        continue
    full = os.path.join(VID_DIR, f)
    end_frame = get_frame_count(full) - 1
    rows.append({
        "video": f,
        "start_frame": 0,
        "end_frame": end_frame if end_frame > 0 else 999999,
        "exercise_type": label,
        "form_label": "good"   
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(df)} rows")


