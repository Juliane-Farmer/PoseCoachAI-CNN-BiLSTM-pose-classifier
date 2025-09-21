import os, argparse, glob, numpy as np, pandas as pd, torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from train_coach_cnn import (CNNBiLSTM, load_npz, seed_everything,make_grouped_stratified_by_video, make_grouped_split_all_classes,)

def resolve_model_path(model_path):
    if model_path and model_path.lower() != "auto":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path not found: {model_path}")
        return model_path
    candidates = glob.glob(os.path.join("model", "*.pt"))
    if not candidates:
        raise FileNotFoundError("No .pt files found under 'model/'.")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def default_out_prefix(out_prefix, model_path):
    if out_prefix:
        return out_prefix
    name = os.path.splitext(os.path.basename(model_path))[0]
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", name)

def get_split(X, y_type, videos, grouped=True, strategy="stratified", val_ratio=0.35, seed=42):
    from sklearn.model_selection import StratifiedShuffleSplit
    if grouped:
        if strategy == "stratified":
            tr, va = make_grouped_stratified_by_video(y_type, videos, val_ratio, seed)
        else:
            tr, va, _ = make_grouped_split_all_classes(X, y_type, videos, seed, val_ratio)
    else:
        (tr, va), = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed).split(X, y_type)
    return tr, va

def resolve_summary_csv(data_npz, summary_csv_cli):
    if summary_csv_cli and os.path.exists(summary_csv_cli):
        return summary_csv_cli
    base = os.path.splitext(data_npz)[0]
    candidates = [
        f"{base}_summary.csv",
        os.path.join(os.path.dirname(data_npz), "windows_phase2_norm_summary.csv"),
        os.path.join(os.path.dirname(data_npz), "windows_phase2_summary.csv"),]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Missing summary CSV. Tried: {', '.join(candidates)}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate Phase-2 model: confusion matrix + report")
    ap.add_argument("--model_path", default="auto")
    ap.add_argument("--data_npz", default="dataset/windows_phase2_norm.npz")
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--grouped_split", type=int, default=1)
    ap.add_argument("--grouped_strategy", choices=["stratified","random"], default="stratified")
    ap.add_argument("--val_ratio", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cnn_channels", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=256)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()
    seed_everything(args.seed)
    model_path = resolve_model_path(args.model_path)
    out_prefix = default_out_prefix(args.out_prefix, model_path)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    print(f"Using model: {model_path}")
    print(f"Output prefix: {out_prefix}")

    X, y_type, y_form, feat_cols, type_names, _ = load_npz(args.data_npz)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    summary_csv = resolve_summary_csv(args.data_npz, args.summary_csv)
    meta = pd.read_csv(summary_csv)

    if len(meta) != len(X):
        n = min(len(meta), len(X))
        print(f"[warn] summary rows ({len(meta)}) != windows ({len(X)}); trimming both to {n}.")
        meta = meta.iloc[:n].reset_index(drop=True)
        X = X[:n]
        y_type = y_type[:n]
        if y_form is not None:
            y_form = y_form[:n]
    videos = meta["video"].values
    train_idx, val_idx = get_split(
        X, y_type, videos,
        grouped=bool(args.grouped_split),
        strategy=args.grouped_strategy,
        val_ratio=args.val_ratio,
        seed=args.seed,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_type = int(y_type.max()+1)
    model = CNNBiLSTM(
        in_feats=X.shape[2], cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
        dropout=args.dropout, num_type=n_type, num_form=None).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    xb = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits, _ = model(torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0))
        preds = logits.argmax(1).cpu().numpy()
    y_true = y_type[val_idx]
    cm = confusion_matrix(y_true, preds, labels=np.arange(n_type))
    rep = classification_report(y_true, preds, target_names=type_names if type_names else None, zero_division=0)
    row_labels = [f"true_{t}" for t in (type_names or range(n_type))]
    col_labels = [f"pred_{t}" for t in (type_names or range(n_type))]
    pd.DataFrame(cm, index=row_labels, columns=col_labels).to_csv(f"{out_prefix}_cm.csv", index=True)
    with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_labels = type_names if type_names else [str(i) for i in range(n_type)]
    plt.xticks(np.arange(n_type), tick_labels, rotation=45, ha="right")
    plt.yticks(np.arange(n_type), tick_labels)
    thresh = cm.max()/2.0 if cm.max() > 0 else 0.5
    for i in range(n_type):
        for j in range(n_type):
            v = cm[i, j]
            plt.text(j, i, str(v), ha="center", va="center",
                     color=("white" if v > thresh else "black"), fontsize=8)
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{out_prefix}_cm.png", dpi=160)
    print("Saved:", f"{out_prefix}_cm.csv", f"{out_prefix}_report.txt", f"{out_prefix}_cm.png")

if __name__ == "__main__":
    main()
