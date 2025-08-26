import time
import sys, os
from pathlib import Path
from collections import deque, defaultdict
import threading
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import torch
import mediapipe as mp

st.set_page_config(page_title="PoseCoachAI — Live")
ROOT = Path(__file__).resolve().parents[1] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from model.train_coach_cnn import CNNBiLSTM, load_npz
except ModuleNotFoundError:
    from train_coach_cnn import CNNBiLSTM, load_npz


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
VIS_THRESH = 0.5
POSE_LM = mp.solutions.pose.PoseLandmark
ANGLE_SPECS = {
    "right_elbow_angle": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_elbow_angle":  ("LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST"),
    "right_knee_angle":  ("RIGHT_HIP",      "RIGHT_KNEE",  "RIGHT_ANKLE"),
    "left_knee_angle":   ("LEFT_HIP",       "LEFT_KNEE",   "LEFT_ANKLE"),
    "right_hip_angle":   ("RIGHT_SHOULDER", "RIGHT_HIP",   "RIGHT_KNEE"),
    "left_hip_angle":    ("LEFT_SHOULDER",  "LEFT_HIP",    "LEFT_KNEE"),
    "right_shoulder_abd": ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    "left_shoulder_abd":  ("LEFT_HIP",  "LEFT_SHOULDER",  "LEFT_ELBOW"),}
BASE_FEATS = list(ANGLE_SPECS.keys()) + ["trunk_tilt"]

def _get_xyzv(lm, name):
    idx = getattr(POSE_LM, name)
    pt = lm[idx]
    return np.array([pt.x, pt.y, pt.z], dtype=np.float32), float(pt.visibility)

def _angle(a, b, c):
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def _safe_joint_angle(lm, A, B, C):
    a, va = _get_xyzv(lm, A); b, vb = _get_xyzv(lm, B); c, vc = _get_xyzv(lm, C)
    if min(va, vb, vc) < VIS_THRESH:
        return np.nan
    return _angle(a, b, c)

def _trunk_tilt_deg(lm):
    ls, vs = _get_xyzv(lm, "LEFT_SHOULDER")
    rs, vsr = _get_xyzv(lm, "RIGHT_SHOULDER")
    lh, vh = _get_xyzv(lm, "LEFT_HIP")
    rh, vhr = _get_xyzv(lm, "RIGHT_HIP")
    if min(vs, vsr, vh, vhr) < VIS_THRESH:
        return np.nan
    mid_sh = (ls + rs) / 2.0
    mid_hip = (lh + rh) / 2.0
    torso = mid_sh - mid_hip
    n = np.linalg.norm(torso)
    if n == 0: return np.nan
    torso = torso / n
    vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cosang = float(np.clip(np.dot(torso, vertical), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

class OnlineFeatureBuilder:
    """Adds *_diff and *_ma5 online, frame-by-frame."""
    def __init__(self, base_names, ma_window=5):
        self.base = base_names
        self.prev = defaultdict(lambda: None)
        self.hist = {n: deque(maxlen=ma_window) for n in base_names}
    def push(self, feats):
        out = {}
        for n in self.base:
            v = feats.get(n, np.nan)
            p = self.prev[n]
            if p is None or np.isnan(p) or np.isnan(v):
                diff = 0.0 if not np.isnan(v) else np.nan
            else:
                diff = float(v - p)
            self.prev[n] = v
            if not np.isnan(v):
                self.hist[n].append(float(v))
            ma = float(np.mean(self.hist[n])) if len(self.hist[n]) else (0.0 if not np.isnan(v) else np.nan)
            out[n] = float(v) if not np.isnan(v) else np.nan
            out[f"{n}_diff"] = diff
            out[f"{n}_ma5"] = ma
        return out

def _vec_from_feats(frame_feats, feat_cols):
    v = np.zeros((len(feat_cols),), dtype=np.float32)
    for i, name in enumerate(feat_cols):
        x = frame_feats.get(name, np.nan)
        v[i] = 0.0 if np.isnan(x) else float(x)
    return v

def _simple_tip(exercise, last_raw):
    if not last_raw:
        return None
    tilt = last_raw.get("trunk_tilt", 0.0) or 0.0
    ex = (exercise or "").lower()
    if ex == "squat":
        knee = min(last_raw.get("left_knee_angle", 180.0) or 180.0,
                   last_raw.get("right_knee_angle", 180.0) or 180.0)
        if knee > 110: return "Go deeper (~90° knees)."
        if tilt > 25:  return "Keep chest up."
    if ex in ("pushup", "push-up", "push_up"):
        elbow = min(last_raw.get("left_elbow_angle", 180.0) or 180.0,
                    last_raw.get("right_elbow_angle", 180.0) or 180.0)
        if elbow > 110: return "Lower further (~90° elbows)."
        if tilt > 15:   return "Keep body in a straight line."
    if ex in ("pullup", "pull-ups", "pull ups"):
        return "Full hang; drive elbows down; chin over bar."
    if ex in ("russian twists", "russian twist"):
        return "Rotate shoulders, not just arms; keep torso tall."
    return None

@st.cache_resource
def load_engine(model_path=str(ROOT / "model/phase2_cnn_bilstm_v9.pt"),
                train_npz=str(ROOT / "dataset/windows_phase2_norm.npz"),
                cnn_channels=256, lstm_hidden=256, lstm_layers=2, dropout=0.5):
    X, y_type, y_form, feat_cols, type_names, form_names = load_npz(train_npz)
    z = np.load(train_npz, allow_pickle=True)
    mu, sd = z["mu"], z["sd"]
    seq_len = int(z["seq_len"][0]) if "seq_len" in z else 200
    stride_pct = float(z["stride_pct"][0]) if "stride_pct" in z else 0.10
    z.close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM(in_feats=len(feat_cols),
                      cnn_channels=cnn_channels,
                      lstm_hidden=lstm_hidden,
                      lstm_layers=lstm_layers,
                      dropout=dropout,
                      num_type=int(y_type.max()+1),
                      num_form=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return {
        "model": model, "device": device, "feat_cols": list(feat_cols),
        "type_names": type_names or [str(i) for i in range(int(y_type.max()+1))],
        "mu": mu, "sd": sd, "seq_len": seq_len,
        "stride": max(1, int(round(seq_len * stride_pct)))}

ENGINE = load_engine()
REQUIRED_BASE = set()
for n in ENGINE["feat_cols"]:
    b = n
    if b.endswith("_diff"): b = b[:-5]
    if b.endswith("_ma5"):  b = b[:-4]
    REQUIRED_BASE.add(b)
REQUIRED_BASE.add("trunk_tilt")

class OverlayProcessor(VideoProcessorBase):
    def __init__(self, selected="(auto)", speak=False):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.builder = OnlineFeatureBuilder(list(REQUIRED_BASE), ma_window=5)
        self.frames = deque(maxlen=ENGINE["seq_len"])
        self.frame_idx = 0
        self.last_raw = {}
        self.pose_present = deque(maxlen=30) 
        self.ready = False
        self.last_label = "???"
        self.last_conf = 0.0
        self.last_tip = None
        self.status = "warming_up"
        self._prev_label = None
        self.selected = selected
        self.speak = speak
        self._tts_lock = threading.Lock()
        self._last_spoken = ""
        self._t_prev = time.time()

    def set_selected(self, name: str):
        self.selected = name

    def set_speak(self, enabled: bool):
        self.speak = enabled

    def _say_async(self, text: str):
        def run():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        with self._tts_lock:
            if text and text != self._last_spoken:
                self._last_spoken = text
                threading.Thread(target=run, daemon=True).start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        has_pose = bool(res.pose_landmarks)
        self.pose_present.append(has_pose)
        feats_raw = {}
        if has_pose:
            lm = res.pose_landmarks.landmark
            for base in REQUIRED_BASE:
                if base == "trunk_tilt":
                    feats_raw["trunk_tilt"] = _trunk_tilt_deg(lm)
                elif base in ANGLE_SPECS:
                    A, B, C = ANGLE_SPECS[base]
                    feats_raw[base] = _safe_joint_angle(lm, A, B, C)
        for k in REQUIRED_BASE:
            feats_raw.setdefault(k, np.nan)
        self.last_raw = feats_raw
        feats = self.builder.push(feats_raw)
        vec = _vec_from_feats(feats, ENGINE["feat_cols"])
        self.frames.append(vec)
        self.frame_idx += 1

        if len(self.frames) >= ENGINE["seq_len"]:
            self.ready = True

        enough_pose = (len(self.pose_present) >= 10 and
                       (sum(self.pose_present) / len(self.pose_present)) >= 0.6)

        fps = None
        if self.ready and enough_pose and (self.frame_idx % ENGINE["stride"] == 0):
            X = np.stack(self.frames, axis=0)[-ENGINE["seq_len"]:]
            X = (X - ENGINE["mu"]) / ENGINE["sd"]
            xb = torch.from_numpy(X[None, ...].astype(np.float32)).to(ENGINE["device"])
            with torch.no_grad():
                logits, _ = ENGINE["model"](xb)
                probs = torch.softmax(logits, dim=1)
                c, p = torch.max(probs, dim=1)
            self.last_conf = float(c.item())
            self.last_label = ENGINE["type_names"][int(p.item())]
            self.status = "ok"
            chosen = self.last_label if self.selected == "(auto)" else self.selected
            tip = _simple_tip(chosen, self.last_raw)

            if self.speak and self.last_conf >= 0.90:
                if tip:
                    self._say_async(tip)
                elif self._prev_label != self.last_label:
                    self._say_async(self.last_label)
            self.last_tip = tip
            self._prev_label = self.last_label
            fps = 1.0 / max(1e-6, (time.time() - self._t_prev))
            self._t_prev = time.time()
        else:
            if self.ready and not enough_pose:
                self.status = "no_pose"
        y = 22
        color_status = (0, 255, 255) if self.status == "no_pose" else (255, 255, 255)
        cv2.putText(img, f"status: {self.status}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2); y += 22
        cv2.putText(img, f"pred: {self.last_label}  conf: {self.last_conf*100:.1f}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 22
        if fps:
            cv2.putText(img, f"fps: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 22
        if self.last_tip:
            cv2.putText(img, f"tip: {self.last_tip}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2); y += 22

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("PoseCoachAI — Real-time Coach")
col1, col2 = st.columns(2)
with col1:
    selected_ex = st.selectbox(
        "Select exercise",
        options=ENGINE["type_names"] + ["(auto)"],
        index=len(ENGINE["type_names"]),)
with col2:
    speak_enabled = st.toggle("Voice tips", value=False, help="Say tips/labels when confident")

ctx = webrtc_streamer(
    key="posecoach",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width": 640, "height": 360}, "audio": False},
    video_processor_factory=lambda: OverlayProcessor(selected=selected_ex, speak=speak_enabled),
    rtc_configuration=RTC_CONFIGURATION,)

if ctx.video_processor is not None:
    ctx.video_processor.set_selected(selected_ex)
    ctx.video_processor.set_speak(speak_enabled)
