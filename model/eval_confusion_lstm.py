import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from train_exercise_classifier import load_and_segment, ExerciseLSTM
from sklearn.model_selection import train_test_split

X, y, input_size, num_classes, label_names = load_and_segment(
    "dataset/exercise_detection/exercise_angles.csv",
    seq_len=200,
    stride_pct=0.25,
    augment=False,
    seed=42)

_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = ExerciseLSTM(
    input_size=input_size,
    hidden_size=256,         
    num_classes=num_classes,
    num_layers=2,            
    dropout=0.5)
model.load_state_dict(torch.load("model/exercise_lstm02.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X_val, dtype=torch.float32))
    preds = logits.argmax(dim=1).numpy()

labels = sorted(pd.read_csv("dataset/exercise_detection/exercise_angles.csv")["Label"].unique())
rev_map = {i: label for i, label in enumerate(label_names)}
y_true = [rev_map[i] for i in y_val]
y_pred = [rev_map[i] for i in preds]

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=labels))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels))
