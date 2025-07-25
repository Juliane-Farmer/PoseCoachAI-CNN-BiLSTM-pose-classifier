import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_and_segment(csv_path, seq_len, stride_pct, augment, seed):
    np.random.seed(seed)

    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c.lower() == 'label']
    if label_cols:
        label_col = label_cols[0]
    else:
        obj = df.select_dtypes(include=['object']).columns.tolist()
        label_col = next((c for c in obj if c.lower() not in ('side','pose')), obj[0])
    print(f"Using '{label_col}' as label column.")
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[features].values.astype(np.float32)
    labels = df[label_col].values

    unique_labels = sorted(np.unique(labels))
    mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
    inv_map = {i: lbl for lbl, i in mapping.items()}
    num_classes = len(unique_labels)
    print(f"Found {num_classes} classes: {unique_labels}")

    stride = int(seq_len * (1 - stride_pct))
    X_seq, y_seq = [], []
    for start in range(0, len(data) - seq_len + 1, stride):
        window = data[start:start+seq_len]
        lbl = labels[start]
        idx = mapping[lbl]
        X_seq.append(window)
        y_seq.append(idx)
        if augment:
            n_copies = 3 if lbl in ('Squats', 'Russian twists') else 1
            for _ in range(n_copies):
                noisy = window + np.random.normal(0, 5.0, window.shape)
                X_seq.append(noisy)
                y_seq.append(idx)

    X = np.stack(X_seq)
    y = np.array(y_seq, dtype=np.int64)
    print(f"Segmented into {len(X)} sequences (seq_len={seq_len}, stride={stride_pct}, augment={augment})")
    return X, y, len(features), num_classes, unique_labels


class ExerciseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = os.path.join(args.data_dir, 'exercise_angles.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    X, y, input_size, num_classes, label_names = load_and_segment(
        csv_path, args.seq_len, args.stride, args.augment, args.seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val,   dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = ExerciseLSTM(
        input_size, args.hidden_size, num_classes,
        args.num_layers, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        avg_val = val_loss / len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"Train Loss: {avg_train:.4f} "
            f"Val Loss: {avg_val:.4f} Val Acc: {val_acc:.4f}")
        scheduler.step(avg_val)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output_model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print("Early stopping triggered.")
                break

    print(f"Best Val Acc: {best_val_acc:.4f}, model saved to {args.output_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',      type=str,   default='dataset/exercise_detection')
    parser.add_argument(
        '--seq_len',       type=int,   default=200)
    parser.add_argument(
        '--stride',        type=float, default=0.1)
    parser.add_argument(
        '--augment',       action='store_true')
    parser.add_argument(
        '--hidden_size',   type=int,   default=256)
    parser.add_argument(
        '--num_layers',    type=int,   default=2)
    parser.add_argument(
        '--dropout',       type=float, default=0.5)
    parser.add_argument(
        '--weight_decay',  type=float, default=1e-4)
    parser.add_argument(
        '--epochs',        type=int,   default=60)
    parser.add_argument(
        '--batch_size',    type=int,   default=32)
    parser.add_argument(
        '--lr',            type=float, default=1e-3)
    parser.add_argument(
        '--early_stop',    type=int,   default=8)
    parser.add_argument(
        '--seed',          type=int,   default=42)
    parser.add_argument(
        '--output_model',  type=str,   default='model/exercise_lstm02.pt')
    args = parser.parse_args()
    train(args)
