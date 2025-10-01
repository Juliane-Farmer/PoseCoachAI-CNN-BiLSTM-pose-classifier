from pathlib import Path
import numpy as np
import torch

def load_engine(root: Path):
    npz_path = root / "dataset" / "windows_phase2_norm.npz"
    model_path = root / "model" / "phase2_cnn_bilstm_v10.pt"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset npz: {npz_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]; y_type = z["y_type"]; feat_cols = z["feat_cols"]; mu, sd = z["mu"], z["sd"]
    type_names = z["type_names"] if "type_names" in z.files else np.array([str(i) for i in range(int(y_type.max()+1))])
    seq_len = int(z["seq_len"][0]) if "seq_len" in z.files else 200
    z.close()
    in_feats = X.shape[2]
    try:
        from model.train_coach_cnn import CNNBiLSTM
    except ModuleNotFoundError:
        from train_coach_cnn import CNNBiLSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM(
        in_feats=in_feats, cnn_channels=256, lstm_hidden=256, lstm_layers=2, dropout=0.5, num_type=int(y_type.max()+1), num_form=None).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, int(seq_len), in_feats, dtype=torch.float32, device=device)
        _ = model(dummy)  


    return {
        "model": model,
        "device": device,
        "mu": mu.astype("float32"),
        "sd": sd.astype("float32"),
        "feat_cols": list(map(str, feat_cols)),
        "type_names": list(map(str, type_names)),
        "seq_len": int(seq_len),}
