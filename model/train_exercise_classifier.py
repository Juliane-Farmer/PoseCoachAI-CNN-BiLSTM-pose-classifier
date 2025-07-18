import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def load_and_segment(csv_path: str, seq_len: int):
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c.lower() == 'label']
    if label_cols:
        label_col = label_cols[0]
    else:
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        obj_cols = [c for c in obj_cols if c.lower() not in ('side', 'pose')]
        if not obj_cols:
            raise ValueError("No suitable label column found in CSV")
        label_col = obj_cols[0]
    print(f"Using '{label_col}' as label column.")

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not feature_cols:
        raise ValueError("No numeric feature columns found in CSV")
    data = df[feature_cols].values.astype(np.float32)
    labels = df[label_col].values

    unique_labels = sorted(np.unique(labels))
    mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Found {num_classes} classes: {unique_labels}")

    total_rows = data.shape[0]
    n_seq = total_rows // seq_len
    if n_seq == 0:
        raise ValueError(f"Not enough rows ({total_rows}) for seq_len={seq_len}")
    trimmed = n_seq * seq_len
    data = data[:trimmed]
    labels = labels[:trimmed]
    X = data.reshape(n_seq, seq_len, len(feature_cols))
    y_raw = labels.reshape(n_seq, seq_len)[:, 0]
    y = np.array([mapping[val] for val in y_raw], dtype=np.int64)
    return X, y, len(feature_cols), num_classes

class ExerciseLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def train(args):
    data_dir = args.data_dir or os.path.join(os.getcwd(), 'dataset', 'exercise_detection')
    csv_path = os.path.join(data_dir, 'exercise_angles.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Cannot find exercise_angles.csv in {data_dir}")
    print(f"Loading and segmenting data from {csv_path}...")

    X, y, input_size, num_classes = load_and_segment(csv_path, args.seq_len)
    print(f"Segmented into {X.shape[0]} sequences; input_size={input_size}, classes={num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val,   dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExerciseLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=num_classes,
        num_layers=args.num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch}/{args.epochs}  Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f}")
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',    type=str, default=None,
        help='Path to folder containing exercise_angles.csv')
    parser.add_argument('--seq_len',     type=int,   default=100)
    parser.add_argument('--hidden_size', type=int,   default=64)
    parser.add_argument('--num_layers',  type=int,   default=1)
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument(
        '--output_model', type=str, default='model/exercise_lstm.pt',
        help='Output path for the trained model')
    args = parser.parse_args()
    train(args)
