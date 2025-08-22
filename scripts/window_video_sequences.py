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

def _select_feature_cols(df: pd.DataFrame) -> list:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in num_cols if c not in META_COLS]
    drop_tokens = {"label", "exercise_type", "form_label", "start_frame", "end_frame"}
    feat_cols = [c for c in feat_cols if c.lower() not in drop_tokens]
    if len(feat_cols) <= 6:
        print(f"[warn] Only {len(feat_cols)} numeric features detected. "
              f"Check that compute_angles.py produced *_diff and *_ma5 columns.")
    return feat_cols

def _per_video_fill(df_vid: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    out = df_vid.copy()
    out[feat_cols] = out[feat_cols].interpolate(method="linear", limit_direction="both")
    med = out[feat_cols].median(numeric_only=True)
    out[feat_cols] = out[feat_cols].fillna(med)
    out_vals = np.nan_to_num(out[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out[feat_cols] = pd.DataFrame(out_vals, columns=feat_cols, index=df_vid.index)
    return out

def _encode_labels(manifest: pd.DataFrame):
    type_names = sorted(manifest["exercise_type"].dropna().unique().tolist())
    type2id = {t: i for i, t in enumerate(type_names)}
    form_names = sorted(manifest["form_label"].dropna().unique().tolist()) \
                 if "form_label" in manifest.columns else ["good"]
    form2id = {t: i for i, t in enumerate(form_names)}
    return type2id, type_names, form2id, form_names

def main():
    args = parse_args()
    if not os.path.isfile(args.angles_csv):
        raise FileNotFoundError(f"Missing angles CSV: {args.angles_csv}")
    if not os.path.isfile(args.manifest_csv):
        raise FileNotFoundError(f"Missing manifest CSV: {args.manifest_csv}")

    angles = pd.read_csv(args.angles_csv)
    manifest = pd.read_csv(args.manifest_csv)

    if "video" not in angles.columns or "frame" not in angles.columns:
        raise ValueError("angles_csv must contain 'video' and 'frame' columns.")
    required = {"video", "start_frame", "end_frame", "exercise_type"}
    if not required.issubset(set(manifest.columns)):
        raise ValueError(f"manifest_csv must contain {required}")

    angles = angles.sort_values(["video", "frame"]).reset_index(drop=True)
    feat_cols = _select_feature_cols(angles)
    type2id, type_names, form2id, form_names = _encode_labels(manifest)
    X_list, y_type_list, y_form_list, rows = [], [], [], []
    seq_len = args.seq_len
    stride = max(1, int(round(seq_len * args.stride_pct)))

    for _, seg in manifest.iterrows():
        vid = seg["video"]
        vtype = seg["exercise_type"]
        vform = seg.get("form_label", "good")
        start_f = int(seg.get("start_frame", 0))
        end_f   = int(seg.get("end_frame", 999_999))

        dfv = angles[angles["video"] == vid].copy()
        if dfv.empty:
            print(f"[warn] No rows in angles for video='{vid}'. Skipping.")
            continue
        dfv = dfv[(dfv["frame"] >= start_f) & (dfv["frame"] <= end_f)]
        if len(dfv) < seq_len:
            continue

        dfv = _per_video_fill(dfv, feat_cols)
        n_frames = len(dfv)
        for s in range(0, n_frames - seq_len + 1, stride):
            e = s + seq_len
            Xwin = dfv[feat_cols].iloc[s:e].to_numpy(dtype=np.float32, copy=True)
            Xwin = np.nan_to_num(Xwin, nan=0.0, posinf=0.0, neginf=0.0)
            X_list.append(Xwin)
            y_type_list.append(type2id[vtype])
            y_form_list.append(form2id.get(vform, 0))
            rows.append({
                "video": vid,
                "win_start": int(dfv["frame"].iloc[s]),
                "win_end": int(dfv["frame"].iloc[e-1]),
                "exercise_type": vtype,
                "form_label": vform})
    if not X_list:
        raise RuntimeError("No windows were created. Check manifests and angles CSV.")

    X = np.stack(X_list, axis=0) 
    y_type = np.array(y_type_list, dtype=np.int64)
    y_form = np.array(y_form_list, dtype=np.int64)

    if args.normalize:
        mu = X.reshape(-1, X.shape[-1]).mean(axis=0, dtype=np.float32)
        sd = X.reshape(-1, X.shape[-1]).std(axis=0, dtype=np.float32) + 1e-6
        Xn = (X - mu) / sd
        np.savez_compressed(
            args.norm_out,
            X=Xn, y_type=y_type, y_form=y_form,
            feat_cols=np.array(feat_cols, dtype=object),
            type_names=np.array(type_names, dtype=object),
            form_names=np.array(form_names, dtype=object))
        print(f"Saved normalized windows to {args.norm_out} with X.shape={Xn.shape}")

    np.savez_compressed(
        args.out_npz,
        X=X, y_type=y_type, y_form=y_form,
        feat_cols=np.array(feat_cols, dtype=object),
        type_names=np.array(type_names, dtype=object),
        form_names=np.array(form_names, dtype=object))
    pd.DataFrame(rows).to_csv(args.summary_csv, index=False)

    print(f"Saved windows to {args.out_npz} with X.shape={X.shape}")
    print(f"Saved window summary to {args.summary_csv}")
    print(f"Classes: exercise_type={type_names}  form_label={form_names}")


if __name__ == "__main__":
    main()
