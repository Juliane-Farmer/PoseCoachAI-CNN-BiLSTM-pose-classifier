import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def build_windows(angles_csv, labels_csv, out_npz,seq_len=200, stride_pct=0.1,normalize=False, artifacts_dir="artifacts/phase2"):
    angles_df = pd.read_csv(angles_csv)
    labels_df = pd.read_csv(labels_csv)

    for col in ["video", "frame"]:
        if col not in angles_df.columns:
            raise ValueError(f"Missing '{col}' in {angles_csv}")

    for col in ["video", "exercise_type", "form_label"]:
        if col not in labels_df.columns:
            raise ValueError(f"Missing '{col}' in {labels_csv}")

    feat_cols = [c for c in angles_df.columns if c.endswith("_angle")]
    if not feat_cols:
        raise ValueError("No *_angle columns found. Run compute_angles.py first.")

    scaler = None
    if normalize:
        scaler = StandardScaler()
        angles_df[feat_cols] = scaler.fit_transform(angles_df[feat_cols].values)
        os.makedirs(artifacts_dir, exist_ok=True)
        import pickle
        with open(os.path.join(artifacts_dir, "scaler_phase2.pkl"), "wb") as f:pickle.dump(scaler, f)

    type_enc = LabelEncoder().fit(labels_df["exercise_type"])
    form_enc = LabelEncoder().fit(labels_df["form_label"])

    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "exercise_label_map.json"), "w") as f:
        json.dump({int(i): lbl for i, lbl in enumerate(type_enc.classes_)}, f, indent=2)
    with open(os.path.join(artifacts_dir, "form_label_map.json"), "w") as f:
        json.dump({int(i): lbl for i, lbl in enumerate(form_enc.classes_)}, f, indent=2)

    stride = max(1, int(seq_len * (1 - stride_pct)))
    X, y_type, y_form, metas = [], [], [], []
    angles_df = angles_df.sort_values(["video", "frame"])
    grouped = dict(tuple(angles_df.groupby("video")))
    skipped = 0
    for _, row in labels_df.iterrows():
        vid = row["video"]
        if vid not in grouped:
            print(f"[warn] video '{vid}' not found in {angles_csv}, skipping.")
            continue
        sub = grouped[vid]
        if "start_frame" in labels_df.columns and not pd.isna(row.get("start_frame", np.nan)):
            start = int(row["start_frame"])
        else:
            start = int(sub["frame"].min())

        if "end_frame" in labels_df.columns and not pd.isna(row.get("end_frame", np.nan)):
            end = int(row["end_frame"])
        else:
            end = int(sub["frame"].max())
        seg = sub[(sub["frame"] >= start) & (sub["frame"] <= end)]
        data = seg[feat_cols].values.astype(np.float32)

        if len(data) < seq_len:
            skipped += 1
            continue

        etype = type_enc.transform([row["exercise_type"]])[0]
        ftype = form_enc.transform([row["form_label"]])[0]

        for s in range(0, len(data) - seq_len + 1, stride):
            win = data[s:s+seq_len]
            X.append(win)
            y_type.append(etype)
            y_form.append(ftype)
            metas.append({
                "video": vid,
                "start_frame": int(seg["frame"].iloc[s]),
                "end_frame": int(seg["frame"].iloc[s+seq_len-1]),
                "exercise_type": row["exercise_type"],
                "form_label": row["form_label"]})
    if not X:
        raise RuntimeError("No windows generated. Check seq_len / labels ranges.")

    X = np.stack(X)                      
    y_type = np.array(y_type, dtype=np.int64)
    y_form = np.array(y_form, dtype=np.int64)
    
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, X=X, y_type=y_type, y_form=y_form,feat_cols=np.array(feat_cols, dtype=object))
    summary_csv = os.path.splitext(out_npz)[0] + "_summary.csv"
    pd.DataFrame(metas).to_csv(summary_csv, index=False)

    print(f"Saved windows to {out_npz}")
    print(f"Feature columns: {len(feat_cols)}")
    print(f"Windows: {len(X)}  | seq_len: {seq_len}  | stride: {stride}")
    if skipped:
        print(f"Skipped {skipped} short segments (<{seq_len} frames).")

def main():
    ap = argparse.ArgumentParser(description="Window video angles into labeled sequences for Phase 2.")
    ap.add_argument("--angles_csv", default="dataset/video_angles.csv")
    ap.add_argument("--labels_csv", default="dataset/video_labels.csv")
    ap.add_argument("--out_npz",    default="dataset/windows_phase2.npz")
    ap.add_argument("--seq_len",    type=int, default=200)
    ap.add_argument("--stride_pct", type=float, default=0.1)
    ap.add_argument("--normalize",  action="store_true")
    ap.add_argument("--artifacts_dir", default="artifacts/phase2")
    args = ap.parse_args()

    build_windows(args.angles_csv, args.labels_csv, args.out_npz,
                  seq_len=args.seq_len, stride_pct=args.stride_pct,
                  normalize=args.normalize, artifacts_dir=args.artifacts_dir)


if __name__ == "__main__":
    main()
