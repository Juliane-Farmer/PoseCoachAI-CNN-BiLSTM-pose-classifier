import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report


class CNNBiLSTM(nn.Module):
    def __init__(self, in_feats, cnn_channels=256, lstm_hidden=256,
                 lstm_layers=2, dropout=0.5, num_type=5, num_form=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_feats, cnn_channels, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=0.0 if lstm_layers == 1 else dropout)
        self.head_type = nn.Linear(lstm_hidden * 2, num_type)
        self.head_form = nn.Linear(lstm_hidden * 2, num_form) if (num_form and num_form > 1) else None

    def forward(self, x):
        x = x.permute(0, 2, 1)                             
        x = self.dropout(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(torch.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)                           
        _, (hn, _) = self.lstm(x)                           
        h = torch.cat([hn[-2], hn[-1]], dim=1)           
        logits_type = self.head_type(self.dropout(h))
        logits_form = self.head_form(self.dropout(h)) if self.head_form is not None else None
        return logits_type, logits_form


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y_type = data["y_type"]
    y_form = data["y_form"]
    feat_cols = list(data["feat_cols"])
    type_names = list(data["type_names"]) if "type_names" in data else None
    form_names = list(data["form_names"]) if "form_names" in data else None
    return X, y_type, y_form, feat_cols, type_names, form_names


def augment_batch(x, time_shift=5, time_mask_prob=0.10, time_mask_len=12, feat_noise_std=0.05):
    """Temporal augs on (B,T,F) numpy array."""
    B, T, F = x.shape
    out = x.copy()
    if time_shift > 0:
        shifts = np.random.randint(-time_shift, time_shift + 1, size=B)
        for i, s in enumerate(shifts):
            if s != 0:
                out[i] = np.roll(out[i], s, axis=0)
    if time_mask_prob > 0:
        for i in range(B):
            if np.random.rand() < time_mask_prob and T > time_mask_len:
                start = np.random.randint(0, T - time_mask_len)
                out[i, start:start + time_mask_len, :] = 0.0
    if feat_noise_std > 0:
        out += np.random.normal(0, feat_noise_std, size=out.shape).astype(out.dtype)
    return out


def class_weights_from_labels(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  
        self.gamma = gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        idx = torch.arange(logits.size(0), device=logits.device)
        pt = p[idx, target]
        loss = -((1 - pt) ** self.gamma) * logp[idx, target]
        if self.alpha is not None:
            at = self.alpha[target]
            loss = loss * at
        return loss.mean()


def make_grouped_split_all_classes(X, y, groups, seed, val_ratio=0.35, max_tries=200):
    """Try many grouped splits, prefer one where val covers all classes."""
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(y))
    best = None
    best_unique = -1
    train_size = max(0.0, min(1.0, 1.0 - float(val_ratio)))
    for _ in range(max_tries):
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=rng.randint(0, 10_000))
        train_idx, val_idx = next(gss.split(X, y, groups))
        u = len(np.unique(y[val_idx]))
        if u == n_classes:
            return train_idx, val_idx, True
        if u > best_unique:
            best_unique = u
            best = (train_idx, val_idx)
    print(f"[warn] Could not include all {n_classes} classes in val; using split with {best_unique}.")
    return best[0], best[1], False


def make_grouped_stratified_by_video(y_type, videos, val_ratio=0.35, seed=42):
    """Stratify at the VIDEO level using each video's majority class."""
    vids, inv = np.unique(videos, return_inverse=True)
    vid_labels = np.zeros(len(vids), dtype=int)
    for i in range(len(vids)):
        cls_counts = np.bincount(y_type[inv == i])
        vid_labels[i] = cls_counts.argmax()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    (train_vid_idx, val_vid_idx), = sss.split(vids, vid_labels)
    train_mask = np.isin(inv, train_vid_idx)
    val_mask   = np.isin(inv, val_vid_idx)
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    return train_idx, val_idx


def make_stratified_split(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    (train_idx, val_idx), = sss.split(X, y)
    return train_idx, val_idx

def apply_preset(args):
    if args.preset is None:
        return args

    if args.preset == "baseline":
        args.data_npz        = args.data_npz or "dataset/windows_phase2_norm.npz"
        args.grouped_split   = 1
        args.grouped_strategy= "stratified"
        args.val_ratio       = 0.35
        args.class_balance   = False
        args.focal_loss      = False
        args.gamma           = 2.0
        args.augment         = False
        args.time_shift      = 5
        args.time_mask_prob  = 0.10
        args.time_mask_len   = 12
        args.feat_noise_std  = 0.05

    elif args.preset == "ce_balanced":
        args.data_npz        = args.data_npz or "dataset/windows_phase2_norm.npz"
        args.grouped_split   = 1
        args.grouped_strategy= "stratified"
        args.val_ratio       = 0.35
        args.class_balance   = True
        args.focal_loss      = False
        args.gamma           = 2.0
        args.augment         = False

    elif args.preset == "v7b":
        args.data_npz        = args.data_npz or "dataset/windows_phase2_norm.npz"
        args.grouped_split   = 1
        args.grouped_strategy= "stratified"
        args.val_ratio       = 0.35
        args.class_balance   = True
        args.focal_loss      = True
        args.gamma           = 2.0
        args.augment         = True
        args.time_shift      = 5
        args.time_mask_prob  = 0.10
        args.time_mask_len   = 12
        args.feat_noise_std  = 0.05
    else:
        print(f"[warn] Unknown preset '{args.preset}', ignoring.")
    return args


def train(args):
    seed_everything(args.seed)
    args = apply_preset(args)
    X, y_type, y_form, feat_cols, type_names, form_names = load_npz(args.data_npz)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Sanity: bad values in X = {np.isnan(X).sum() + np.isinf(X).sum()}")
    summary_csv = args.summary_csv or os.path.splitext(args.data_npz)[0] + "_summary.csv"
    if not os.path.exists(summary_csv):
        alt = os.path.join(os.path.dirname(args.data_npz), "windows_phase2_summary.csv")
        if os.path.exists(alt):
            summary_csv = alt
        else:
            raise FileNotFoundError(f"Could not find summary CSV. Tried: {summary_csv} and {alt}")

    meta = pd.read_csv(summary_csv)
    assert len(meta) == len(X), "summary rows must match number of windows"
    groups = meta["video"].values
    unique_form = np.unique(y_form)
    use_form = len(unique_form) > 1

    if args.grouped_split:
        n_vids = len(np.unique(groups))
        n_cls  = len(np.unique(y_type))
        val_vids = int(round(n_vids * args.val_ratio))
        if val_vids < n_cls:
            print(f"[warn] val videos ({val_vids}) < num classes ({n_cls}); increase --val_ratio or add videos.")
        if args.grouped_strategy == "stratified":
            train_idx, val_idx = make_grouped_stratified_by_video(y_type, groups, args.val_ratio, args.seed)
            ok = len(np.unique(y_type[val_idx])) == n_cls
        else:
            train_idx, val_idx, ok = make_grouped_split_all_classes(
                X, y_type, groups, seed=args.seed, val_ratio=args.val_ratio)
    else:
        train_idx, val_idx = make_stratified_split(X, y_type, seed=args.seed)
        ok = len(np.unique(y_type[val_idx])) == len(np.unique(y_type))

    print(f"X shape: {X.shape}  (windows={X.shape[0]}, seq_len={X.shape[1]}, features={X.shape[2]})")
    print("Train class counts:", np.bincount(y_type[train_idx], minlength=int(y_type.max()+1)))
    print("Val   class counts:", np.bincount(y_type[val_idx],  minlength=int(y_type.max()+1)))
    if type_names:
        print("Classes (type_names):", type_names)
    print("Example features:", feat_cols[:10], "...")

    n_type_classes = int(y_type.max() + 1)
    type_weights = class_weights_from_labels(y_type[train_idx], num_classes=n_type_classes)
    form_weights = class_weights_from_labels(y_form[train_idx], num_classes=int(y_form.max() + 1)) if use_form else None

    def make_loader(ids, shuffle=True, weighted=False, weights_vec=None):
        Xt = torch.tensor(X[ids], dtype=torch.float32)
        yt = torch.tensor(y_type[ids], dtype=torch.long)
        if use_form:
            yf = torch.tensor(y_form[ids], dtype=torch.long)
            ds = TensorDataset(Xt, yt, yf)
        else:
            ds = TensorDataset(Xt, yt)
        if weighted and weights_vec is not None:
            sample_w = weights_vec.numpy()[yt.numpy()]
            sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
            return DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    if args.class_balance:
        train_loader = make_loader(train_idx, weighted=True, weights_vec=type_weights)
    else:
        train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM(
        in_feats=X.shape[2],
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_type=n_type_classes,
        num_form=(int(y_form.max()+1) if use_form else None)).to(device)

    smoothing = 0.0 if args.focal_loss else args.label_smoothing
    if args.focal_loss:
        alpha = type_weights.to(device) if args.class_balance else None
        type_criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
    else:
        if args.class_balance:
            type_criterion = nn.CrossEntropyLoss(weight=type_weights.to(device), label_smoothing=smoothing)
        else:
            type_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    ce_form = None
    if use_form and model.head_form is not None:
        if args.class_balance and form_weights is not None:
            ce_form = nn.CrossEntropyLoss(weight=form_weights.to(device), label_smoothing=smoothing)
        else:
            ce_form = nn.CrossEntropyLoss(label_smoothing=smoothing)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", patience=3, factor=0.5)

    best_f1 = 0.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optim.zero_grad()
            if use_form:
                xb, yb_type, yb_form = batch
            else:
                xb, yb_type = batch

            if args.augment:
                xb_np = xb.numpy()
                xb_np = augment_batch(
                    xb_np,
                    time_shift=args.time_shift,
                    time_mask_prob=args.time_mask_prob,
                    time_mask_len=args.time_mask_len,
                    feat_noise_std=args.feat_noise_std)
                xb = torch.from_numpy(xb_np)

            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            xb = xb.to(device); yb_type = yb_type.to(device)
            logits_type, logits_form = model(xb)
            loss = type_criterion(logits_type, yb_type)
            if use_form and ce_form is not None and logits_form is not None:
                yb_form = yb_form.to(device)
                loss = loss + args.lambda_form * ce_form(logits_form, yb_form)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optim.step()
            total_loss += loss.item()

        model.eval()
        preds_type, truths_type = [], []
        preds_form, truths_form = [], []
        with torch.no_grad():
            for batch in val_loader:
                if use_form:
                    xb, yb_type, yb_form = [t.to(device) for t in batch]
                else:
                    xb, yb_type = [t.to(device) for t in batch]
                xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
                lt, lf = model(xb)
                pt = lt.argmax(1).cpu().numpy()
                preds_type.extend(pt); truths_type.extend(yb_type.cpu().numpy())
                if use_form and lf is not None:
                    pf = lf.argmax(1).cpu().numpy()
                    preds_form.extend(pf); truths_form.extend(yb_form.cpu().numpy())

        f1_type = f1_score(truths_type, preds_type, average="macro", zero_division=0)
        acc_type = accuracy_score(truths_type, preds_type)

        if use_form and len(preds_form) > 0:
            f1_form = f1_score(truths_form, preds_form, average="macro", zero_division=0)
            acc_form = accuracy_score(truths_form, preds_form)
            print(f"Epoch {epoch}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  "
                  f"type: acc={acc_type:.3f} f1={f1_type:.3f}  "
                  f"form: acc={acc_form:.3f} f1={f1_form:.3f}")
        else:
            print(f"Epoch {epoch}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  "
                  f"type: acc={acc_type:.3f} f1={f1_type:.3f}")

        sched.step(f1_type)
        if f1_type > best_f1 + 1e-4:
            best_f1 = f1_type; no_improve = 0
            torch.save(model.state_dict(), args.out_model)
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print("Early stopping.")
                break

    print("\nFinal validation report (exercise type):")
    print(classification_report(truths_type, preds_type, zero_division=0))
    if use_form and len(preds_form) > 0:
        print("\nFinal validation report (form label):")
        print(classification_report(truths_form, preds_form, zero_division=0))
    print(f"\nBest model saved to {args.out_model}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=[None, "baseline", "ce_balanced", "v7b"], default="baseline")

    ap.add_argument("--data_npz", default=None)  
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--out_model", default="model/phase2_cnn_bilstm_v9.pt")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--early_stop", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cnn_channels", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=256)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)

    ap.add_argument("--lambda_form", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--grouped_split", type=int, default=1,help="1: grouped by video (no leakage); 0: stratified by window.")
    ap.add_argument("--grouped_strategy", choices=["random","stratified"], default="stratified", help="How to pick val videos when grouped_split=1.")
    ap.add_argument("--val_ratio", type=float, default=0.35, help="Validation ratio for grouped splits (increase so val videos ≥ #classes).")

    ap.add_argument("--class_balance", action="store_true", help="Use class-balanced sampling / loss.")
    ap.add_argument("--focal_loss", action="store_true", help="Use focal loss for TYPE head.")
    ap.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma.")

    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--time_shift", type=int, default=5)
    ap.add_argument("--time_mask_prob", type=float, default=0.10)
    ap.add_argument("--time_mask_len", type=int, default=12)
    ap.add_argument("--feat_noise_std", type=float, default=0.05)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())


