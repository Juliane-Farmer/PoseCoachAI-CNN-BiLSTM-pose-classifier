import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

class CNNBiLSTM(nn.Module):
    def __init__(self, in_feats, cnn_channels=128, lstm_hidden=128,
                 lstm_layers=1, dropout=0.5, num_type=5, num_form=None):
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
        self.num_form  = num_form
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

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['X'], data['y_type'], data['y_form'], list(data['feat_cols'])

def make_grouped_split(X, y_type, groups, seed, tries=100):
    """Try many splits and pick one with the most classes in val."""
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(y_type))
    best = None
    best_unique = -1
    for _ in range(tries):
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8,
                                random_state=rng.randint(0, 10_000))
        train_idx, val_idx = next(gss.split(X, y_type, groups))
        u = len(np.unique(y_type[val_idx]))
        if u == n_classes:
            return train_idx, val_idx
        if u > best_unique:
            best_unique = u
            best = (train_idx, val_idx)
    print(f"[warn] Could not include all {n_classes} classes in val; using split with {best_unique}.")
    return best

def class_weights_from_labels(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

def augment_batch(x, time_shift=10, time_mask_prob=0.2, time_mask_len=12, feat_noise_std=0.02):
    """Simple temporal augs on (B,T,F) numpy array."""
    B, T, F = x.shape
    out = x.copy()
    if time_shift > 0:
        shifts = np.random.randint(-time_shift, time_shift + 1, size=B)
        for i, s in enumerate(shifts):
            if s != 0:
                out[i] = np.roll(out[i], s, axis=0)
    if time_mask_prob > 0:
        for i in range(B):
            if np.random.rand() < time_mask_prob:
                start = np.random.randint(0, max(1, T - time_mask_len))
                out[i, start:start + time_mask_len, :] = 0.0
    if feat_noise_std > 0:
        out += np.random.normal(0, feat_noise_std, size=out.shape).astype(out.dtype)
    return out

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    X, y_type, y_form, feat_cols = load_npz(args.data_npz)
    summary_csv = args.summary_csv or os.path.splitext(args.data_npz)[0] + "_summary.csv"
    meta = pd.read_csv(summary_csv)
    assert len(meta) == len(X), "summary rows must match number of windows"
    groups = meta['video'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Sanity: bad values in X = {np.isnan(X).sum() + np.isinf(X).sum()}")
    seed_everything(args.seed)

    unique_form = np.unique(y_form)
    use_form = len(unique_form) > 1
    train_idx, val_idx = make_grouped_split(X, y_type, groups, seed=args.seed)

    print(f"X shape: {X.shape}  (windows={X.shape[0]}, seq_len={X.shape[1]}, features={X.shape[2]})")
    print("Train class counts:", np.bincount(y_type[train_idx], minlength=len(np.unique(y_type))))
    print("Val   class counts:", np.bincount(y_type[val_idx],  minlength=len(np.unique(y_type))))
    print("Example features:", feat_cols[:10], "...")

    type_weights = class_weights_from_labels(y_type[train_idx], num_classes=len(np.unique(y_type)))
    form_weights = class_weights_from_labels(y_form[train_idx], num_classes=len(unique_form)) if use_form else None

    def make_loader(ids, shuffle=True, weighted=False):
        Xt = torch.tensor(X[ids], dtype=torch.float32)
        yt = torch.tensor(y_type[ids], dtype=torch.long)
        if use_form:
            yf = torch.tensor(y_form[ids], dtype=torch.long)
            ds = TensorDataset(Xt, yt, yf)
        else:
            ds = TensorDataset(Xt, yt)
        if weighted:
            sample_w = type_weights.numpy()[yt.numpy()]
            sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
            return DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, weighted=True)
    val_loader   = make_loader(val_idx, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNBiLSTM(
        in_feats=X.shape[2],
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_type=len(np.unique(y_type)),
        num_form=len(unique_form) if use_form else None).to(device)

    ce_type = nn.CrossEntropyLoss(weight=type_weights.to(device), label_smoothing=args.label_smoothing)
    ce_form = nn.CrossEntropyLoss(weight=form_weights.to(device), label_smoothing=args.label_smoothing) if (use_form and form_weights is not None) else None
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5)
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            if use_form:
                xb, yb_type, yb_form = batch
            else:
                xb, yb_type = batch

            if args.augment:
                xb_np = xb.numpy()
                xb_np = augment_batch(
                    xb_np, time_shift=args.time_shift,
                    time_mask_prob=args.time_mask_prob,
                    time_mask_len=args.time_mask_len,
                    feat_noise_std=args.feat_noise_std)
                xb = torch.from_numpy(xb_np)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)

            xb = xb.to(device); yb_type = yb_type.to(device)
            logits_type, logits_form = model(xb)
            loss = ce_type(logits_type, yb_type)
            if use_form and logits_form is not None:
                yb_form = yb_form.to(device)
                loss = loss + args.lambda_form * ce_form(logits_form, yb_form)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
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

        f1_type = f1_score(truths_type, preds_type, average='macro', zero_division=0)
        acc_type = accuracy_score(truths_type, preds_type)

        if use_form and len(preds_form) > 0:
            f1_form = f1_score(truths_form, preds_form, average='macro', zero_division=0)
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
    ap.add_argument("--data_npz", default="dataset/windows_phase2.npz")
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--out_model", default="model/phase2_cnn_bilstm_angles_v4.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cnn_channels", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=256)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lambda_form", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--early_stop", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--time_shift", type=int, default=10)
    ap.add_argument("--time_mask_prob", type=float, default=0.2)
    ap.add_argument("--time_mask_len", type=int, default=12)
    ap.add_argument("--feat_noise_std", type=float, default=0.02)
    return ap.parse_args()

if __name__ == "__main__":
    train(parse_args())


