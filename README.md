# PoseCoachAI — Real-Time AI Exercise Coach

PoseCoachAI analyzes your workout in real time using pose estimation and deep learning. It extracts human pose keypoints (MediaPipe), computes biomechanical angles and temporal features, classifies exercises, and speaks actionable coaching tips to improve form and reduce injury risk.

## Features
- Live camera analysis with Streamlit + WebRTC  
- Exercise classification with CNN-BiLSTM  
- Form feedback: depth, lockout, stance, trunk tilt, rhythm  
- Voice tips (Windows SAPI locally; optional browser speech)  
- Session summaries with download + clear actions   

## Tech StackV
| Layer | Tech |
|---|---|
| Pose Estimation | MediaPipe |
| ML Model | CNN-BiLSTM (PyTorch) |
| UI | Streamlit + streamlit-webrtc |
| Audio | pyttsx3/SAPI (Windows), optional browser TTS |
| Data | NumPy, Pandas |
| Eval | Scikit-learn |

## Folder Structure
```
PoseCoachAI/
│
├── ui/
│   ├── app.py
│   └── overlay_processor.py
│
├── posecoach/
│   ├── inference/
│   │   └── engine.py
│   ├── coaching/
│   │   ├── coaching_rules.py
│   │   └── tip_aggregator.py
│   └── tts.py
│
├── model/
│   ├── train_coach_cnn.py
│   ├── eval_confusion_cnn.py
│   ├── phase2_cnn_bilstm_v*.pt
│   ├── performances.txt
│   └── best.json
│
├── dataset/
│   ├── videos/                # ignored
│   ├── keypoints/             # ignored
│   ├── windows_phase2_norm.npz
│   └── .gitkeep
│
├── scripts/
│   ├── compute_angles.py
│   ├── data_vis.py
│   ├── dataset_download.py
│   ├── extract_keypoints.py
│   ├── fetch_videos.py
│   ├── make_manifest.py
│   └── window_video_sequences.py
│
├── outputs/
│   ├── session_logs/
│   └── .gitkeep
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Getting Started

### 1) Clone
```bash
git clone https://github.com/Juliane-Farmer/PoseCoachAI.git
cd PoseCoachAI
```

### 2) Create and activate a virtual environment
**Windows (cmd)**
```cmd
python -m venv PoseCoachvenv
PoseCoachvenv\Scripts\activate.bat
```
**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4) Add model and dataset artifacts
- Place a trained checkpoint in `model/`, e.g. `phase2_cnn_bilstm_v10.pt`
- Ensure `dataset/windows_phase2_norm.npz` exists

### 5) Run the app
```bash
streamlit run ui/app.py
```
Open http://localhost:8501, allow the camera, and select an exercise.

## Usage Tips
- Wear tight or fitted clothing; avoid loose garments that hide joints  
- Toggle “Voice tips” to hear coaching  
- Perform a short set and click **End set ▶ Speak summary** for a recap  
- Download or clear the last summary beneath the camera      

## Development
Data prep scripts live in `scripts/`. Train/evaluate with files under `model/`.

## Troubleshooting
- Ensure `dataset/windows_phase2_norm.npz` and at least one `model/phase2_cnn_bilstm_v*.pt` are present  
- Use Chrome/Edge and allow camera permissions for `localhost`  
- On Windows, local voice tips use SAPI via pyttsx3  
- Set `POSECOACH_TTS=browser` to use browser speech synthesis  

## Acknowledgments
MediaPipe for robust pose estimation, Streamlit and streamlit-webrtc for the UI and camera streaming, and PyTorch for training and inference.
