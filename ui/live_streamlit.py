import time
import sys
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _build_logger():
    logger = logging.getLogger("posecoach")
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_dir = ROOT / "outputs" / "session_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"live_{int(time.time())}.log"
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info(f"Logging to {log_path}")
    return logger

LOGGER = _build_logger()
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
SMOOTH_K = 6
WARMUP_PAD = True
IDLE_MOVEMENT_THRESH = 0.4
IDLE_HOLD_STEPS = 3
RESET_UNKNOWN_STEPS = 3
UNKNOWN_THRESH = 0.50
DEFAULT_SPEAK_GATE = 0.80
BASE_MARGIN = 0.05
EARLY_GUARD_STEPS = 15
QUALITY_NEUTRALIZE_MAX = 0.35
QUALITY_COVERAGE_MIN = 0.60
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

ALL_JOINTS = sorted(set(sum([list(v) for v in REQUIRED_JOINTS.values()], [])))

def margin_needed(p_top: float) -> float:
    return max(BASE_MARGIN, 0.5 * (1.0 - float(p_top)))

def steps_needed(p_top: float) -> int:
    return 3 if p_top < 0.70 else (2 if p_top < 0.85 else 1)

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

class OnlineFeatureBuilder:
    def __init__(self, base_names, ma_window=5):
        self.base = list(base_names)
        self.prev = defaultdict(lambda: None)
        self.hist = {n: deque(maxlen=ma_window) for n in base_names}
    def push(self, feats):
        out = {}
        for n in self.base:
            v = feats.get(n, np.nan)
            p = self.prev[n]
            if p is None or np.isnan(p) or np.isnan(v):
                diff = np.nan
            else:
                diff = float(v - p)
            self.prev[n] = v
            if not np.isnan(v):
                self.hist[n].append(float(v))
            ma = float(np.mean(self.hist[n])) if len(self.hist[n]) > 0 else np.nan
            out[n] = float(v) if not np.isnan(v) else np.nan
            out[f"{n}_diff"] = diff
            out[f"{n}_ma5"]  = ma
        return out

@st.cache_resource
def load_engine():
    npz_path = ROOT / "dataset" / "windows_phase2_norm.npz"
    model_path = ROOT / "model" / "phase2_cnn_bilstm_v9.pt"
    if not npz_path.exists():
        st.error(f"Missing: {npz_path}")
        st.stop()
    if not model_path.exists():
        st.error(f"Missing: {model_path}")
        st.stop()
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]; y_type = z["y_type"]; feat_cols = z["feat_cols"]
    mu, sd = z["mu"], z["sd"]
    type_names = z["type_names"] if "type_names" in z.files else np.array([str(i) for i in range(int(y_type.max()+1))])
    seq_len = int(z["seq_len"][0]) if "seq_len" in z.files else 200
    z.close()
    in_feats = X.shape[2]
    try:
        from model.train_coach_cnn import CNNBiLSTM
    except ModuleNotFoundError:
        from train_coach_cnn import CNNBiLSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM(in_feats=in_feats, cnn_channels=256, lstm_hidden=256, lstm_layers=2,dropout=0.5, num_type=int(y_type.max()+1), num_form=None).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return {"model": model, "device": device, "mu": mu, "sd": sd, "feat_cols": list(map(str, feat_cols)),
            "type_names": list(map(str, type_names)), "seq_len": int(seq_len)}

ENGINE = load_engine()

def _base_of(n):
    if n.endswith("_diff"): return n[:-5]
    if n.endswith("_ma5"): return n[:-4]
    return n

REQUIRED_BASE = sorted({_base_of(n) for n in ENGINE["feat_cols"]} | {"trunk_tilt"})
COMPUTABLE_BASE = set(ANGLE_SPECS.keys()) | {"trunk_tilt"}
UNCOMP_BASE = sorted(b for b in REQUIRED_BASE if b not in COMPUTABLE_BASE)
if UNCOMP_BASE:
    LOGGER.warning("Neutralizing uncomputable bases: %s", ", ".join(UNCOMP_BASE))

def _vec_from_feats(frame_feats, feat_cols, mu_by_index, uncomputable_bases):
    v = np.zeros((len(feat_cols),), dtype=np.float32)
    neutralized = 0
    for i, name in enumerate(feat_cols):
        base = _base_of(name)
        if base in uncomputable_bases:
            v[i] = mu_by_index[i]; neutralized += 1; continue
        x = frame_feats.get(name, np.nan)
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            v[i] = mu_by_index[i]; neutralized += 1
        else:
            v[i] = float(x)
    return v, neutralized / max(1, len(feat_cols))

def _simple_tip(exercise, last_raw):
    if not last_raw: return None
    tilt = last_raw.get("trunk_tilt", 0.0) or 0.0
    ex = (exercise or "").lower()
    if ex in ("squat", "squats"):
        knee_min = min(last_raw.get("left_knee_angle", 180), last_raw.get("right_knee_angle", 180))
        if knee_min > 160: return "Bend knees more"
        if tilt > 25: return "Keep chest up"
    elif ex in ("jumping jacks",):
        l = last_raw.get("left_shoulder_abd", 90); r = last_raw.get("right_shoulder_abd", 90)
        if min(l, r) < 60: return "Raise arms higher"
    elif ex in ("pushup", "push-ups"):
        hip = min(last_raw.get("left_hip_angle", 180), last_raw.get("right_hip_angle", 180))
        if hip < 160: return "Keep hips aligned"
    return None

def visible_fraction(lm, joint_names, thresh=0.5):
    if lm is None: return 0.0
    vis = []
    for name in joint_names:
        _, v = _get_xyzv(lm, name)
        vis.append(v >= thresh)
    return float(np.mean(vis)) if vis else 0.0

class OverlayProcessor(VideoProcessorBase):
    def __init__(self, selected="(auto)", speak=False, speak_gate=DEFAULT_SPEAK_GATE):
        self.selected = selected
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.prob_hist = deque(maxlen=SMOOTH_K)
        self.feat_hist = deque(maxlen=SMOOTH_K)
        self.ready = False
        self.pose_present = deque(maxlen=15)
        self.last_label = "unknown"; self.last_conf = 0.0
        self.cand_label = None; self.cand_steps = 0
        self.unknown_steps = 0
        self.last_raw = {}
        self.debug_top3 = ""
        self.idle_steps = 0
        self.speak = speak; self.speak_gate = speak_gate
        self._tts_lock = threading.Lock(); self._last_spoken = None
        self._t_prev = time.time()
        self._unk_boost_until = 0.0
        self._cooldown = 0
        self._builder = None
        self._neut_frac = 0.0
        self._knee_flex = 0
        self.status = "init"
        self._coverage = 0.0
        self._lower_cov = 0.0

    def set_selected(self, s): self.selected = s
    def set_speak(self, enabled, gate=DEFAULT_SPEAK_GATE):
        self.speak = enabled; self.speak_gate = gate

    def _say_async(self, text):
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
        fps = None
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        has_pose = bool(res.pose_landmarks)
        self.pose_present.append(has_pose)
        lm = res.pose_landmarks.landmark if has_pose else None
        self._coverage = visible_fraction(lm, ALL_JOINTS, thresh=0.5) if has_pose else 0.0
        self._lower_cov = visible_fraction(lm, ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"], thresh=0.5) if has_pose else 0.0
        squat_vis = visible_fraction(lm, REQUIRED_JOINTS.get("squat", []), thresh=0.5) if has_pose else 0.0
        squat_hint = squat_vis >= 0.5
        feats_raw = {}
        if has_pose:
            for base in REQUIRED_BASE:
                if base == "trunk_tilt":
                    feats_raw["trunk_tilt"] = _trunk_tilt_deg(lm)
                elif base in ANGLE_SPECS:
                    A, B, C = ANGLE_SPECS[base]
                    feats_raw[base] = _safe_joint_angle(lm, A, B, C)
        for b in REQUIRED_BASE:
            feats_raw.setdefault(b, np.nan)
        builder = self._builder
        if builder is None:
            self._builder = OnlineFeatureBuilder(REQUIRED_BASE, ma_window=5)
            builder = self._builder
        feats = builder.push(feats_raw)
        self.last_raw = feats_raw.copy()
        knee_min_now = min(self.last_raw.get("left_knee_angle", 180), self.last_raw.get("right_knee_angle", 180))
        if knee_min_now < 150:
            self._knee_flex += 1
        else:
            self._knee_flex = max(0, self._knee_flex - 1)
        strong_squat_hint = (self._knee_flex >= 6)
        mu_by_index = {i: float(ENGINE["mu"][i]) for i,_ in enumerate(ENGINE["feat_cols"])}
        v, neut_frac = _vec_from_feats(feats, ENGINE["feat_cols"], mu_by_index, set(UNCOMP_BASE))
        self._neut_frac = neut_frac
        self.feat_hist.append(v)
        enough_pose = np.mean(list(self.pose_present)) >= 0.5
        quality_ok = (self._coverage >= QUALITY_COVERAGE_MIN) and (self._neut_frac <= QUALITY_NEUTRALIZE_MAX)

        if not self.ready:
            if (WARMUP_PAD and len(self.feat_hist) >= SMOOTH_K) or len(self.feat_hist) >= ENGINE["seq_len"]:
                self.ready = True
            else:
                self.status = "need_full_body" if not enough_pose else "warming"

        if self.ready and enough_pose and quality_ok:
            X_raw = np.stack(list(self.feat_hist), axis=0)
            if len(self.feat_hist) < ENGINE["seq_len"]:
                pad = ENGINE["seq_len"] - len(self.feat_hist)
                mu_row = ENGINE["mu"].astype(np.float32)
                X_raw = np.vstack([np.tile(mu_row, (pad, 1)), X_raw])
            if len(self.feat_hist) >= 11:
                recent = np.diff(np.stack(list(self.feat_hist)[-11:], axis=0), axis=0)
                mov_score = float(np.mean(np.abs(recent)))
            else:
                mov_score = 0.0
            if mov_score < IDLE_MOVEMENT_THRESH:
                self.idle_steps += 1
            else:
                self.idle_steps = 0
            X = (X_raw - ENGINE["mu"]) / ENGINE["sd"]
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

            if selected_key != "(auto)" and selected_key in {n.lower(): i for i, n in enumerate(ENGINE["type_names"])}:
                idx = {n.lower(): i for i, n in enumerate(ENGINE["type_names"])}[selected_key]
                self.last_label = ENGINE["type_names"][idx]
                self.last_conf = float(avg_probs[idx])
            else:
                top = int(avg_probs.argmax())
                second = int(np.argsort(avg_probs)[-2]) if avg_probs.size > 1 else top
                avg_max = float(avg_probs[top]); avg_second = float(avg_probs[second])
                eff_unknown = UNKNOWN_THRESH + (0.05 if time.time() < getattr(self, "_unk_boost_until", 0) else 0.0)
                candidate = "unknown" if avg_max < eff_unknown else ENGINE["type_names"][top]

                if self.idle_steps >= IDLE_HOLD_STEPS and not squat_hint:
                    candidate = "unknown"
                    self._cooldown = getattr(self, "_cooldown", 0) + 1
                else:
                    self._cooldown = max(0, getattr(self, "_cooldown", 0) - 1)

                if len(self.prob_hist) < EARLY_GUARD_STEPS:
                    cand_l = candidate.lower()
                    if cand_l in ("russian twists", "russian twist"):
                        tilt = float(self.last_raw.get("trunk_tilt", 0.0) or 0.0)
                        if not (avg_max >= eff_unknown + 0.20 and tilt < 20.0):
                            candidate = "unknown"

                if getattr(self, "_cooldown", 0) > 0 and candidate != "unknown":
                    candidate = self.last_label if self.last_label != "unknown" else "unknown"

                SQUAT_IDX = next((i for i, n in enumerate(ENGINE["type_names"]) if n.lower() in ("squat", "squats")), None)
                if (self._lower_cov >= 0.6 and self._neut_frac <= 0.30) and (squat_hint or strong_squat_hint) and SQUAT_IDX is not None:
                    p_squat = float(avg_probs[SQUAT_IDX])
                    if p_squat >= 0.40 or (p_squat >= avg_max - 0.10 and p_squat >= 0.30):
                        candidate = ENGINE["type_names"][SQUAT_IDX]
                        avg_max = p_squat

                if self.last_label == "unknown" and candidate != "unknown" and avg_max >= (UNKNOWN_THRESH + 0.10):
                    self.last_label = candidate
                    self.last_conf = avg_max
                    stable_changed = True
                    self.cand_label, self.cand_steps = None, 0
                else:
                    if candidate != self.last_label:
                        if candidate == self.cand_label:
                            self.cand_steps += 1
                        else:
                            self.cand_label = candidate
                            self.cand_steps = 1
                        need_steps = steps_needed(avg_max)
                        need_margin = margin_needed(avg_max)
                        if self.cand_steps >= need_steps and (candidate == "unknown" or (avg_max - avg_second) >= need_margin):
                            self.last_label = candidate
                            self.last_conf = avg_max
                            stable_changed = True
                            self.cand_label, self.cand_steps = None, 0
                    else:
                        self.cand_label, self.cand_steps = None, 0

                if self.last_label == "unknown":
                    self.unknown_steps += 1
                else:
                    self.unknown_steps = 0
                if self.unknown_steps >= RESET_UNKNOWN_STEPS:
                    self.prob_hist.clear()
                    self.cand_label, self.cand_steps = None, 0
                    self._unk_boost_until = time.time() + 1.5

            self.status = "ok"
            shown_ex = self.last_label if selected_key == "(auto)" else self.selected
            tip = _simple_tip(shown_ex, self.last_raw)
            self.last_tip = tip if self.last_label != "unknown" else None
            if self.speak and self.last_label != "unknown" and (stable_changed or self.last_conf >= self.speak_gate):
                self._say_async(self.last_tip or self.last_label)
            fps = 1.0 / max(1e-6, (time.time() - self._t_prev))
            self._t_prev = time.time()
        else:
            self.status = "need_full_body" if not enough_pose else ("low_quality" if not quality_ok else "warming")
            self.last_label = "unknown"
            self.last_conf = 0.0
            self.prob_hist.clear()
            self.cand_label, self.cand_steps = None, 0

        y = 22
        color_status = (0, 165, 255) if self.status in ("no_pose", "need_full_body", "low_quality") else (255, 255, 255)
        cv2.putText(img, f"status: {self.status}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2); y += 22
        conf_txt = f"{self.last_conf*100:.1f}%" if self.last_label != "unknown" else "--"
        cv2.putText(img, f"pred: {self.last_label}  conf: {conf_txt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 22
        if getattr(self, "debug_top3", ""):
            cv2.putText(img, f"top3: {self.debug_top3}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2); y += 22
        if fps is not None:
            cv2.putText(img, f"fps: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 22
        cv2.putText(img, f"neutralized: {int(self._neut_frac*100)}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2); y += 22
        cv2.putText(img, f"coverage: {int(self._coverage*100)}%  lower: {int(self._lower_cov*100)}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2); y += 22
        try:
            idx_map = {n.lower(): i for i, n in enumerate(ENGINE["type_names"])}
            if 'avg_probs' in locals():
                p_rt = float(avg_probs[idx_map.get("russian twists", idx_map.get("russian twist", -1))]) if idx_map.get("russian twists", idx_map.get("russian twist", -1)) not in (None, -1) else 0.0
                p_sq = float(avg_probs[idx_map.get("squat", idx_map.get("squats", -1))]) if idx_map.get("squat", idx_map.get("squats", -1)) not in (None, -1) else 0.0
            else:
                p_rt = p_sq = 0.0
            cv2.putText(img, f"p_squat={p_sq:.2f}  p_rt={p_rt:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2); y += 22
        except Exception:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("PoseCoachAI — Real-time Coach")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    selected_ex = st.selectbox("Select exercise", options=ENGINE["type_names"] + ["(auto)"], index=len(ENGINE["type_names"]))
with col2:
    speak_enabled = st.toggle("Voice tips", value=False)
with col3:
    speak_gate = st.slider("Speak ≥", 0.0, 1.0, DEFAULT_SPEAK_GATE, 0.05)
ctx = webrtc_streamer(
    key="posecoach",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width": 640, "height": 360}, "audio": False},
    video_processor_factory=lambda: OverlayProcessor(selected=selected_ex, speak=speak_enabled, speak_gate=speak_gate),
    rtc_configuration=RTC_CONFIGURATION, )


if ctx.video_processor is not None:
    ctx.video_processor.set_selected(selected_ex)
    ctx.video_processor.set_speak(speak_enabled, gate=speak_gate)



