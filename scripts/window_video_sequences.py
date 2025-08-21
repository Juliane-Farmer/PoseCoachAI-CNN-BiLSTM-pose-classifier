import os
import argparse
import numpy as np
import pandas as pd

META_COLS = {"video", "frame"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles_csv",   default="dataset/video_angles.csv")
    ap.add_argument("--manifest_csv", default="dataset/video_labels.csv")
    ap.add_argument("--out_npz",      default="dataset/windows_phase2.npz")
    ap.add_argument("--summary_csv",  default="dataset/windows_phase2_summary.csv")
    ap.add_argument("--norm_out",     default="dataset/windows_phase2_norm.npz")
    ap.add_argument("--seq_len",      type=int, default=200)
    ap.add_argument("--stride_pct",   type=float, default=0.1)
    ap.add_argument("--normalize",    action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.angles_csv):
        raise FileNotFoundError(f"Missing angles CSV: {args.angles_csv}")
    df = pd.read_csv(args.angles_csv)
    if "video" not in df.columns or "frame" not in df.columns:
        raise ValueError("video_angles.csv must have 'video' and 'frame' columns.")

    feature_cols = [c for c in df.columns if c not in META_COLS and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        raise ValueError("No numeric features found in video_angles.csv.")
    print(f"Using {len(feature_cols)} features:", feature_cols[:12], "..." if len(feature_cols) > 12 else "")

    def repair_group(g):
        g = g.sort_values("frame")
        g[feature_cols] = (
            g[feature_cols]
            .interpolate(limit_direction="both")
            .fillna(g[feature_cols].median()))
        return g

    df = df.groupby("video", group_keys=False).apply(repair_group)

    if args.normalize:
        feat = df[feature_cols].to_numpy(dtype=np.float32)
        mean = np.nanmean(feat, axis=0)
        std  = np.nanstd(feat,  axis=0)
        std[std == 0] = 1.0
        df[feature_cols] = (df[feature_cols] - mean) / std
        np.savez(args.norm_out, mean=mean, std=std, feat_cols=np.array(feature_cols, dtype=object))
        print(f"Saved normalization stats to {args.norm_out}")

    if not os.path.isfile(args.manifest_csv):
        raise FileNotFoundError(f"Missing manifest: {args.manifest_csv}")
    man = pd.read_csv(args.manifest_csv)
    required = {"video", "start_frame", "end_frame", "exercise_type"}
    if not required.issubset(set(man.columns)):
        raise ValueError(f"Manifest must contain columns: {required}")

    if "form_label" not in man.columns:
        man["form_label"] = "unknown"

    type_names = sorted(man["exercise_type"].unique().tolist())
    type_to_idx = {t: i for i, t in enumerate(type_names)}
    form_names = sorted(man["form_label"].unique().tolist())
    form_to_idx = {f: i for i, f in enumerate(form_names)}

    X_list, y_type_list, y_form_list, rows = [], [], [], []
    stride = max(1, int(args.seq_len * (1.0 - args.stride_pct)))

    for _, r in man.iterrows():
        vid = r["video"]
        s0  = int(r["start_frame"])
        e0  = int(r["end_frame"])
        t_i = type_to_idx[r["exercise_type"]]
        f_i = form_to_idx[r["form_label"]]
        g = df[(df["video"] == vid) & (df["frame"] >= s0) & (df["frame"] <= e0)].sort_values("frame")
        if len(g) < args.seq_len:
            continue

        for start in range(0, len(g) - args.seq_len + 1, stride):
            end = start + args.seq_len
            seg = g.iloc[start:end][feature_cols].to_numpy(dtype=np.float32)
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
            if np.isnan(seg).any() or np.isinf(seg).any():
                continue

            X_list.append(seg)
            y_type_list.append(t_i)
            y_form_list.append(f_i)
            rows.append({
                "video": vid,
                "start_frame": int(g.iloc[start]["frame"]),
                "end_frame":   int(g.iloc[end-1]["frame"]),
                "exercise_type": r["exercise_type"],
                "form_label":    r["form_label"]})

    if not X_list:
        raise RuntimeError("No windows were created. Check manifest ranges and seq_len/stride.")

    X = np.stack(X_list, axis=0)
    y_type = np.asarray(y_type_list, dtype=np.int64)
    y_form = np.asarray(y_form_list, dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    np.savez(
        args.out_npz,
        X=X,
        y_type=y_type,
        y_form=y_form,
        feat_cols=np.array(feature_cols, dtype=object),
        type_names=np.array(type_names, dtype=object),
        form_names=np.array(form_names, dtype=object))
    pd.DataFrame(rows).to_csv(args.summary_csv, index=False)

    print(f"Saved windows to {args.out_npz} with X.shape={X.shape}")
    print(f"Saved window summary to {args.summary_csv}")
    print(f"Classes: exercise_type={type_names}  form_label={form_names}")

if __name__ == "__main__":
    main()



