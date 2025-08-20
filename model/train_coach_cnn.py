import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

class CNNBiLSTM(nn.Module):
    def __init__(self, in_feats, cnn_channels=128, lstm_hidden=128,
                 lstm_layers=1, dropout=0.5, num_type=5, num_form=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_feats, cnn_channels, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=0.0 if lstm_layers==1 else dropout)
        self.head_type = nn.Linear(lstm_hidden*2, num_type)
        self.num_form = num_form
        if num_form and num_form > 1:
            self.head_form = nn.Linear(lstm_hidden*2, num_form)
        else:
            self.head_form = None

    def forward(self, x):
        x = x.permute(0,2,1)               
        x = self.dropout(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(torch.relu(self.bn2(self.conv2(x))))
        x = x.permute(0,2,1)             
        _, (hn, _) = self.lstm(x)        
        h = torch.cat([hn[-2], hn[-1]], dim=1) 
        logits_type = self.head_type(self.dropout(h))
        logits_form = self.head_form(self.dropout(h)) if self.head_form is not None else None
        return logits_type, logits_form

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']               
    y_type = data['y_type']
    y_form = data['y_form']
    feat_cols = list(data['feat_cols'])
    return X, y_type, y_form, feat_cols

def train(args):
    X, y_type, y_form, feat_cols = load_npz(args.data_npz)
    summary_csv = args.summary_csv or os.path.splitext(args.data_npz)[0] + "_summary.csv"
    import pandas as pd
    meta = pd.read_csv(summary_csv)
    assert len(meta)==len(X), "summary rows must match number of windows"
    unique_form = np.unique(y_form)
    use_form = len(unique_form) > 1
    groups = meta['video'].values
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=args.seed)
    train_idx, val_idx = next(gss.split(X, y_type, groups))

    def make_loader(ids, shuffle):
        Xt = torch.tensor(X[ids], dtype=torch.float32)
        yt = torch.tensor(y_type[ids], dtype=torch.long)
        if use_form:
            yf = torch.tensor(y_form[ids], dtype=torch.long)
            ds = TensorDataset(Xt, yt, yf)
        else:
            ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader   = make_loader(val_idx, False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNBiLSTM(
        in_feats=X.shape[2],
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_type=len(np.unique(y_type)),
        num_form=len(unique_form) if use_form else None
    ).to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5)

    best_f1 = 0.0; no_improve = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            if use_form:
                xb, yb_type, yb_form = [t.to(device) for t in batch]
            else:
                xb, yb_type = [t.to(device) for t in batch]
            logits_type, logits_form = model(xb)
            loss = ce(logits_type, yb_type)
            if use_form and logits_form is not None:
                loss = loss + args.lambda_form * ce(logits_form, yb_form)
            loss.backward()
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
                lt, lf = model(xb)
                pt = lt.argmax(1).cpu().numpy()
                preds_type.extend(pt); truths_type.extend(yb_type.cpu().numpy())
                if use_form and lf is not None:
                    pf = lf.argmax(1).cpu().numpy()
                    preds_form.extend(pf); truths_form.extend(yb_form.cpu().numpy())

        f1_type = f1_score(truths_type, preds_type, average='macro')
        acc_type = accuracy_score(truths_type, preds_type)
        if use_form and preds_form:
            f1_form = f1_score(truths_form, preds_form, average='macro')
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
    print(classification_report(truths_type, preds_type))
    if use_form and preds_form:
        print("\nFinal validation report (form label):")
        print(classification_report(truths_form, preds_form))
    print(f"\nBest model saved to {args.out_model}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", default="dataset/windows_phase2.npz")
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--out_model", default="model/phase2_cnn_bilstm.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cnn_channels", type=int, default=128)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lambda_form", type=float, default=1.0)
    ap.add_argument("--early_stop", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    train(parse_args())
