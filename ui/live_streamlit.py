import time
import sys
import os
import logging
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

st.set_page_config(page_title="PoseCoachAI — Live", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from model.train_coach_cnn import CNNBiLSTM, load_npz
except ModuleNotFoundError:
    from train_coach_cnn import CNNBiLSTM, load_npz

LOG_DIR = ROOT / "outputs" / "session_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_TS = time.strftime("%Y%m%d-%H%M%S")
LOG_PATH = LOG_DIR / f"live_{SESSION_TS}.log"

def _build_logger():
    logger = logging.getLogger("posecoach")
    if logger.handlers:
        return logger  
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S", )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info(f"Logging to {LOG_PATH}")
    return logger

LOGGER = _build_logger()
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

SMOOTH_K = 8           
HOLD_STEPS = 3        
MARGIN = 0.12         
UNKNOWN_THRESH = 0.45 
DEFAULT_SPEAK_GATE = 0.80

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

REQUIRED_JOINTS = {
    "squat": ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"],
    "squats": ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"],
    "russian twists": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"],
    "russian twist": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"],
    "pushup": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST"],
    "push-ups": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST"],
    "pull ups": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW"],
    "pullups": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW"],
    "jumping jacks": ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_SHOULDER","RIGHT_SHOULDER"],}

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
    if n == 0:
        return np.nan
    torso = torso / n
    vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cosang = float(np.clip(np.dot(torso, vertical), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def visible_fraction(lm, names, thresh=0.5):
    if lm is None:
        return 0.0
    cnt = 0
    for n in names:
        idx = getattr(POSE_LM, n)
        if lm[idx].visibility >= thresh:
            cnt += 1
    return cnt / max(1, len(names))

class OnlineFeatureBuilder:
    """
    Frame-by-frame temporal features:
      - *_diff: first difference
      - *_ma5 : moving average (5)
    Missing base values are imputed with the moving average instead of zeros.
    """
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
                diff = 0.0
            else:
                diff = float(v - p)
            self.prev[n] = v
            if not np.isnan(v):
                self.hist[n].append(float(v))
            ma = float(np.mean(self.hist[n])) if len(self.hist[n]) else 0.0
            base_val = float(v) if not np.isnan(v) else ma
            out[n] = base_val
            out[f"{n}_diff"] = diff
            out[f"{n}_ma5"] = ma
        return out

def _vec_from_feats(frame_feats, feat_cols):
    v = np.zeros((len(feat_cols),), dtype=np.float32)
    for i, name in enumerate(feat_cols):
        v[i] = float(frame_feats.get(name, 0.0))
    return v

def _simple_tip(exercise, last_raw):
    if not last_raw:
        return None
    tilt = last_raw.get("trunk_tilt", 0.0) or 0.0
    ex = (exercise or "").lower()
    if ex == "squat" or ex == "squats":
        knee = min(last_raw.get("left_knee_angle", 180.0) or 180.0,
                   last_raw.get("right_knee_angle", 180.0) or 180.0)
        if knee > 110: return "Go deeper (~90° knees)."
        if tilt > 25:  return "Keep chest up."
    if ex in ("pushup", "push-ups", "push_up"):
        elbow = min(last_raw.get("left_elbow_angle", 180.0) or 180.0,
                    last_raw.get("right_elbow_angle", 180.0) or 180.0)
        if elbow > 110: return "Lower further (~90° elbows)."
        if tilt > 15:   return "Keep body in a straight line."
    if ex in ("pullup", "pull-ups", "pull ups", "pullups"):
        return "Full hang; drive elbows down; chin over bar."
    if ex in ("russian twists", "russian twist"):
        return "Rotate shoulders, not just arms; keep torso tall."
    if ex in ("jumping jacks", "jumpingjack", "jumping-jacks"):
        return "Hands overhead; land softly; keep rhythm."
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
        "type_names": list(type_names) if type_names is not None else [str(i) for i in range(int(y_type.max()+1))],
        "mu": mu, "sd": sd, "seq_len": seq_len,
        "stride": max(1, int(round(seq_len * stride_pct)))}

ENGINE = load_engine()

REQUIRED_BASE = set()
for n in ENGINE["feat_cols"]:
    b = n[:-5] if n.endswith("_diff") else (n[:-4] if n.endswith("_ma5") else n)
    REQUIRED_BASE.add(b)
REQUIRED_BASE.add("trunk_tilt")

NAME2IDX = {str(n).strip().lower(): i for i, n in enumerate(ENGINE["type_names"])}

class OverlayProcessor(VideoProcessorBase):
    def __init__(self, selected="(auto)", speak=False, speak_gate=DEFAULT_SPEAK_GATE):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.builder = OnlineFeatureBuilder(list(REQUIRED_BASE), ma_window=5)
        self.frames = deque(maxlen=ENGINE["seq_len"])
        self.frame_idx = 0
        self.last_raw = {}
        self.pose_present = deque(maxlen=30)  

        self.prob_hist = deque(maxlen=SMOOTH_K)
        self.cand_label = None
        self.cand_steps = 0
        self.ready = False
        self.last_label = "???"
        self.last_conf = 0.0
        self.last_tip = None
        self.debug_top3 = ""
        self.status = "warming_up"
        self._prev_label = None

        self.selected = selected
        self.speak = speak
        self.speak_gate = float(speak_gate)

        self._tts_lock = threading.Lock()
        self._last_spoken = ""

        self._t_prev = time.time()
        LOGGER.info(f"Processor init | selected='{self.selected}' speak={self.speak} gate={self.speak_gate}")

    def set_selected(self, name: str):
        self.selected = name
        LOGGER.info(f"UI change | selected='{self.selected}'")

    def set_speak(self, enabled: bool, gate: float = None):
        just_enabled = (enabled and not self.speak)
        self.speak = enabled
        if gate is not None:
            self.speak_gate = float(gate)
        LOGGER.info(f"UI change | speak={self.speak} gate={self.speak_gate:.2f}")
        if just_enabled:
            self._last_spoken = ""
            self._say_async(self.last_tip or self.last_label or "Coach enabled")

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
                LOGGER.info(f"voice | {text}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        has_pose = bool(res.pose_landmarks)
        self.pose_present.append(has_pose)
        lm = res.pose_landmarks.landmark if has_pose else None
        squat_vis = visible_fraction(lm, REQUIRED_JOINTS.get("squat", []), thresh=0.5) if has_pose else 0.0
        feats_raw = {}
        if has_pose:
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
                logits_type, _ = ENGINE["model"](xb)
                probs = torch.softmax(logits_type, dim=1).cpu().numpy()[0]

            self.prob_hist.append(probs)
            avg_probs = np.mean(np.stack(self.prob_hist, axis=0), axis=0) if len(self.prob_hist) else probs
            top3_idx = avg_probs.argsort()[-3:][::-1]
            self.debug_top3 = " | ".join(f"{ENGINE['type_names'][i]}:{avg_probs[i]*100:.0f}%" for i in top3_idx)

            selected_key = str(self.selected).strip().lower()
            stable_changed = False

            if selected_key != "(auto)" and selected_key in NAME2IDX:
                idx = NAME2IDX[selected_key]
                self.last_label = ENGINE["type_names"][idx]
                self.last_conf = float(avg_probs[idx])
            else:
                top = int(avg_probs.argmax())
                second = int(np.argsort(avg_probs)[-2]) if avg_probs.size > 1 else top
                avg_max = float(avg_probs[top])
                avg_second = float(avg_probs[second])
                candidate = "unknown" if avg_max < UNKNOWN_THRESH else ENGINE["type_names"][top]
                if candidate != self.last_label:
                    if candidate == self.cand_label:
                        self.cand_steps += 1
                    else:
                        self.cand_label = candidate
                        self.cand_steps = 1
                    if self.cand_steps >= HOLD_STEPS and (candidate == "unknown" or (avg_max - avg_second) >= MARGIN):
                        self.last_label = candidate
                        self.last_conf = 0.0 if candidate == "unknown" else avg_max
                        stable_changed = True
                else:
                    self.last_conf = 0.0 if self.last_label == "unknown" else avg_max
                    self.cand_label, self.cand_steps = None, 0

            self.status = "ok"
            shown_ex = self.last_label if selected_key == "(auto)" else self.selected
            tip = _simple_tip(shown_ex, self.last_raw)
            self.last_tip = tip

            LOGGER.info(
                f"status={self.status} | label='{self.last_label}' | conf={self.last_conf:.3f} | "
                f"top3=[{self.debug_top3}] | tip={tip or '-'} | squat_vis={squat_vis:.2f}")

            if self.speak and (stable_changed or self.last_conf >= self.speak_gate):
                self._say_async(tip or self.last_label)
            fps = 1.0 / max(1e-6, (time.time() - self._t_prev))
            self._t_prev = time.time()
        else:
            if self.ready and not enough_pose:
                self.status = "no_pose"

        y = 22
        color_status = (0, 165, 255) if self.status in ("no_pose", "need_full_body") else (255, 255, 255)
        cv2.putText(img, f"status: {self.status}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2); y += 22
        cv2.putText(img, f"pred: {self.last_label}  conf: {self.last_conf*100:.1f}%", (10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 22
        if self.debug_top3:
            cv2.putText(img, f"top3: {self.debug_top3}", (10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2); y += 22
        if fps:
            cv2.putText(img, f"fps: {fps:.1f}", (10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 22
        if self.last_tip:
            cv2.putText(img, f"tip: {self.last_tip}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2); y += 22
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("PoseCoachAI — Real-time Coach")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    selected_ex = st.selectbox(
        "Select exercise",
        options=ENGINE["type_names"] + ["(auto)"],
        index=len(ENGINE["type_names"]),)
with col2:
    speak_enabled = st.toggle("Voice tips", value=False, help="Say tips/labels when confident")
with col3:
    speak_gate = st.slider("Speak ≥", 0.0, 1.0, DEFAULT_SPEAK_GATE, 0.05,help="Confidence threshold for TTS")

ctx = webrtc_streamer(
    key="posecoach",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width": 640, "height": 360}, "audio": False},
    video_processor_factory=lambda: OverlayProcessor(
        selected=selected_ex, speak=speak_enabled, speak_gate=speak_gate),
    rtc_configuration=RTC_CONFIGURATION,)

if ctx.video_processor is not None:
    ctx.video_processor.set_selected(selected_ex)
    ctx.video_processor.set_speak(speak_enabled, gate=speak_gate)



