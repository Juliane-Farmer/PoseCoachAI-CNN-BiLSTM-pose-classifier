import time
import sys
import logging
from pathlib import Path
from collections import deque, defaultdict
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import torch
import mediapipe as mp

from tip_aggregator import TipAggregator
if "tip_agg" not in st.session_state:
    st.session_state.tip_agg = TipAggregator()
if "last_summary_text" not in st.session_state:
    st.session_state.last_summary_text = ""
if "last_summary_ts" not in st.session_state:
    st.session_state.last_summary_ts = 0.0

from tts import TTS
try:
    from phase_detectors import SquatPhaseDetector, JacksPhaseDetector
except Exception:
    SquatPhaseDetector = JacksPhaseDetector = None

from coaching_rules import FormCoach
try:
    from coaching_rules import tips_from_phase_events
except Exception:
    tips_from_phase_events = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _build_logger():
    logger = logging.getLogger("posecoach")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    log_dir = ROOT / "outputs" / "session_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"live_{int(time.time())}.log"
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

LOGGER = _build_logger()

def LOG(msg, level="info"):
    line = f"{time.strftime('%H:%M:%S')} | {level.upper():7s} | {msg}"
    print(line, flush=True)
    getattr(LOGGER, level)(msg)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
SMOOTH_K_DEFAULT = 6
WARMUP_PAD = True
DEFAULT_SPEAK_GATE = 0.30
QUALITY_NEUTRALIZE_MAX = 0.35
QUALITY_COVERAGE_MIN = 0.60
MOV_RECORD_THRESH = 0.25
SPEAK_COOLDOWN = 2.0
VIS_THRESH = 0.5

REST_SECS = 15.0             
NOPOSE_REST_SECS = 15.0         
ACTIVE_MOV_TH = 0.18             
REST_MOV_TH = 0.12               
SUMMARY_COOLDOWN_S = 10.0
AGG_DEDUP_S = 1.0               

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
    "jumping jacks": ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_SHOULDER","RIGHT_SHOULDER"],
    "pull ups": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_HIP","RIGHT_HIP"],}

def _get_xyzv(lm, name):
    idx = getattr(POSE_LM, name); pt = lm[idx]
    return np.array([pt.x, pt.y, pt.z], dtype=np.float32), float(pt.visibility)

def _angle(a, b, c):
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0: return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def _safe_joint_angle(lm, A, B, C):
    a, va = _get_xyzv(lm, A); b, vb = _get_xyzv(lm, B); c, vc = _get_xyzv(lm, C)
    if min(va, vb, vc) < VIS_THRESH: return np.nan
    return _angle(a, b, c)

def _trunk_tilt_deg(lm):
    ls, vs = _get_xyzv(lm, "LEFT_SHOULDER"); rs, vsr = _get_xyzv(lm, "RIGHT_SHOULDER")
    lh, vh = _get_xyzv(lm, "LEFT_HIP"); rh, vhr = _get_xyzv(lm, "RIGHT_HIP")
    if min(vs, vsr, vh, vhr) < VIS_THRESH: return np.nan
    mid_sh = (ls + rs) / 2.0; mid_hip = (lh + rh) / 2.0
    torso = mid_sh - mid_hip; n = np.linalg.norm(torso)
    if n == 0: return np.nan
    torso = torso / n; vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cosang = float(np.clip(np.dot(torso, vertical), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def _simple_tip(exercise, last_raw):
    if not last_raw: return None
    tilt = float(last_raw.get("trunk_tilt", 0.0) or 0.0)
    ex = (exercise or "").lower()
    if ex in ("squat", "squats"):
        lk = float(last_raw.get("left_knee_angle", 180) or 180)
        rk = float(last_raw.get("right_knee_angle", 180) or 180)
        knee_min = min(lk, rk)
        if knee_min > 160: return "Bend knees more"
        if tilt > 25: return "Keep chest up"
    elif ex in ("jumping jacks",):
        l = float(last_raw.get("left_shoulder_abd", 90) or 90)
        r = float(last_raw.get("right_shoulder_abd", 90) or 90)
        if min(l, r) < 60: return "Raise arms higher"
    return None

class OnlineFeatureBuilder:
    def __init__(self, base_names, ma_window=5):
        self.base = list(base_names); self.prev = defaultdict(lambda: None)
        self.hist = {n: deque(maxlen=ma_window) for n in base_names}
    def push(self, feats):
        out = {}
        for n in self.base:
            v = feats.get(n, np.nan); p = self.prev[n]
            diff = np.nan if (p is None or np.isnan(p) or np.isnan(v)) else float(v - p)
            self.prev[n] = v
            if not np.isnan(v): self.hist[n].append(float(v))
            ma = float(np.mean(self.hist[n])) if len(self.hist[n]) > 0 else np.nan
            out[n] = float(v) if not np.isnan(v) else np.nan
            out[f"{n}_diff"] = diff; out[f"{n}_ma5"] = ma
        return out

@st.cache_resource(show_spinner=False)
def load_engine():
    npz_path = ROOT / "dataset" / "windows_phase2_norm.npz"
    model_path = ROOT / "model" / "phase2_cnn_bilstm_v9.pt"
    if not npz_path.exists(): st.error(f"Missing: {npz_path}"); st.stop()
    if not model_path.exists(): st.error(f"Missing: {model_path}"); st.stop()
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]; y_type = z["y_type"]; feat_cols = z["feat_cols"]; mu, sd = z["mu"], z["sd"]
    type_names = z["type_names"] if "type_names" in z.files else np.array([str(i) for i in range(int(y_type.max()+1))])
    seq_len = int(z["seq_len"][0]) if "seq_len" in z.files else 200; z.close()
    in_feats = X.shape[2]
    try:
        from model.train_coach_cnn import CNNBiLSTM
    except ModuleNotFoundError:
        from train_coach_cnn import CNNBiLSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM(in_feats=in_feats, cnn_channels=256, lstm_hidden=256, lstm_layers=2,
                      dropout=0.5, num_type=int(y_type.max()+1), num_form=None).to(device)
    state = torch.load(model_path, map_location=device); model.load_state_dict(state); model.eval()
    return {
        "model": model,
        "device": device,
        "mu": mu,
        "sd": sd,
        "feat_cols": list(map(str, feat_cols)),
        "type_names": list(map(str, type_names)),
        "seq_len": int(seq_len)}

ENGINE = load_engine()
LOG("Logging initialized")

def _base_of(n):
    if n.endswith("_diff"): return n[:-5]
    if n.endswith("_ma5"): return n[:-4]
    return n

REQUIRED_BASE = sorted({_base_of(n) for n in ENGINE["feat_cols"]} | {"trunk_tilt"})
COMPUTABLE_BASE = set(ANGLE_SPECS.keys()) | {"trunk_tilt"}
UNCOMP_BASE = sorted(b for b in REQUIRED_BASE if b not in COMPUTABLE_BASE)
if UNCOMP_BASE:
    LOG(f"Neutralizing uncomputable bases: {', '.join(UNCOMP_BASE)}")

def _vec_from_feats(frame_feats, feat_cols, mu_by_index, uncomputable_bases):
    v = np.zeros((len(feat_cols),), dtype=np.float32); neutralized = 0
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

INFER_SEQ_LEN_DEFAULT = max(40, min(ENGINE["seq_len"], 64))

REPHRASE = {
    "Set feet ~shoulder-width apart": "set a shoulder-width stance",
    "Keep chest up (neutral spine)": "keep your chest up and maintain a neutral spine",
    "Go deeper (hips to at least parallel)": "aim to reach at least parallel depth",
    "Control the descent": "control the lowering phase",
    "Drive up": "drive up powerfully out of the bottom",

    "Get hands fully overhead each rep": "fully extend your arms overhead each rep",
    "Jump wider with your feet": "jump a bit wider with your feet",
    "Keep a steady rhythm": "keep a steady, consistent rhythm",

    "Keep body in one line (brace core)": "keep your body in one line by bracing your core",
    "Lower hips—avoid piking": "avoid piking—keep your hips level",
    "Go deeper—bend elbows more at the bottom": "add a little more depth by bending the elbows further",
    "Press up strong": "press up strongly to lockout",

    "Keep your body in one line (reduce swing)": "reduce swing and keep a straight body line",
    "Pull higher—aim chest toward the bar": "pull a bit higher—think chest to bar",
    "Pull smoothly to the top": "pull smoothly to the top position",

    "Sit tall (avoid rounding the back)": "sit tall and avoid rounding",
    "Rotate shoulders more side-to-side": "rotate your shoulders more side to side",}

class OverlayProcessor(VideoProcessorBase):
    def __init__(self, selected, speak=False, speak_gate=DEFAULT_SPEAK_GATE, show_debug=False,
                 speak_fn=None, infer_seq_len=INFER_SEQ_LEN_DEFAULT, smooth_k=SMOOTH_K_DEFAULT):
        self.selected = selected
        self.show_debug = show_debug
        self.pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.prob_hist = deque(maxlen=int(smooth_k))
        self.feat_hist = deque(maxlen=int(infer_seq_len))

        self.ready = False
        self.pose_present = deque(maxlen=15)
        self.last_label = "unknown"; self.last_conf = 0.0
        self.last_raw = {}; self.debug_top3 = ""
        self.speak = speak; self.speak_gate = speak_gate
        self._t_prev = time.time()
        self._builder = None; self._neut_frac = 0.0
        self.status = "init"; self._coverage = 0.0; self._lower_cov = 0.0
        self._mov_score = 0.0
        self._announced_ready = False
        self._last_tip = None; self._last_tip_t = 0.0
        self.coach = FormCoach()
        self._last_status = None
        self.speak_fn = speak_fn or (lambda _txt: None)

        self.frame_i = 0
        self.squat_pd = SquatPhaseDetector() if SquatPhaseDetector else None
        self.jacks_pd = JacksPhaseDetector() if JacksPhaseDetector else None

        self.agg = TipAggregator()
        self._agg_last_time = {}         
        self.last_active_ts = time.time()
        self.last_pose_ts = time.time()
        self.idle_started_ts = None
        self.last_summary_ts = 0.0

    @property
    def tips_this_set(self):
        return self.agg.total

    def compose_humane_summary(self, k=2, min_count=2):
        if not self.agg.counts:
            return "Nice work—no major corrections this set."
        ex_counts = defaultdict(int)
        for (ex, tip), c in self.agg.counts.items():
            ex_counts[ex] += c
        exercise = max(ex_counts.items(), key=lambda kv: kv[1])[0]
        items = sorted(self.agg.counts.items(), key=lambda kv: (kv[1], self.agg.last_ts[kv[0]]), reverse=True)
        phrases = []
        for (ex, tip), cnt in items:
            if ex != exercise:
                continue
            if cnt < min_count and len(phrases) >= k:
                continue
            phrase = REPHRASE.get(tip, tip.lower())
            phrases.append(phrase)
            if len(phrases) >= k:
                break
        if not phrases and items:
            phrases = [REPHRASE.get(items[0][0][1], items[0][0][1].lower())]
        friendly_ex = exercise.capitalize()
        return f"Here's some feedback from this set of {friendly_ex}: " + "; ".join(phrases) + "."

    def make_summary(self, k=2, min_count=2):
        return self.compose_humane_summary(k=k, min_count=min_count)

    def clear_tips(self):
        self.agg.clear()
        self._agg_last_time.clear()

    def speak_summary(self, k=2, min_count=2):
        if not self.speak:
            return None
        summary = self.make_summary(k=k, min_count=min_count)
        if summary:
            self.speak_fn(summary)
            self.last_summary_ts = time.time()
            st.session_state.last_summary_text = summary
            st.session_state.last_summary_ts = self.last_summary_ts
            self.clear_tips()
            LOG(f"speak_summary | {summary}")
        return summary

    def set_selected(self, s):
        self.selected = s
        self._announced_ready = False
        self._last_tip = None
        self._last_tip_t = 0.0
        LOG(f"ui | selected={self.selected}")

    def set_speak(self, enabled, gate=DEFAULT_SPEAK_GATE):
        self.speak = enabled; self.speak_gate = gate
        LOG(f"ui | speak={self.speak} gate={self.speak_gate:.2f}")

    def set_debug(self, val):
        self.show_debug = bool(val)
        LOG(f"ui | debug={self.show_debug}")

    def set_params(self, infer_seq_len=None, smooth_k=None):
        if infer_seq_len and int(infer_seq_len) != self.feat_hist.maxlen:
            new_len = int(infer_seq_len)
            old = list(self.feat_hist)[-new_len:]
            self.feat_hist = deque(old, maxlen=new_len)
        if smooth_k and int(smooth_k) != self.prob_hist.maxlen:
            new_len = int(smooth_k)
            old = list(self.prob_hist)[-new_len:]
            self.prob_hist = deque(old, maxlen=new_len)

    def _speak_if_ok(self, tip, ignore_conf=False):
        reasons = []
        if not self.speak:
            reasons.append("speak=False")
        if self._coverage < 0.50:
            reasons.append(f"cov={self._coverage:.2f}<0.50")
        if (not ignore_conf) and (self.last_conf < self.speak_gate):
            reasons.append(f"conf={self.last_conf:.2f}<gate={self.speak_gate:.2f}")
        if not tip:
            reasons.append("no-tip")
        if reasons:
            LOG("speak_skip | " + ", ".join(reasons))
            return
        self.speak_fn(tip)
        self._last_tip = tip
        self._last_tip_t = time.time()
        LOG(f"speak | {tip}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_i += 1
        fps = None
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        has_pose = bool(res.pose_landmarks); self.pose_present.append(has_pose)
        lm = res.pose_landmarks.landmark if has_pose else None
        all_joints = sorted(set(sum([list(v) for v in REQUIRED_JOINTS.values()], [])))
        self._coverage = (np.mean([(_get_xyzv(lm, n)[1] >= VIS_THRESH) for n in all_joints]) if has_pose else 0.0)
        self._lower_cov = (np.mean([(_get_xyzv(lm, n)[1] >= VIS_THRESH) for n in REQUIRED_JOINTS["squat"]]) if has_pose else 0.0)
        feats_raw = {}
        if has_pose:
            for base in REQUIRED_BASE:
                if base == "trunk_tilt":
                    feats_raw["trunk_tilt"] = _trunk_tilt_deg(lm)
                elif base in ANGLE_SPECS:
                    A, B, C = ANGLE_SPECS[base]; feats_raw[base] = _safe_joint_angle(lm, A, B, C)
        for b in REQUIRED_BASE:
            feats_raw.setdefault(b, np.nan)

        if self._builder is None:
            self._builder = OnlineFeatureBuilder(REQUIRED_BASE, ma_window=5)
        feats = self._builder.push(feats_raw)
        self.last_raw = feats_raw.copy()
        mu_by_index = {i: float(ENGINE["mu"][i]) for i,_ in enumerate(ENGINE["feat_cols"])}
        v, neut_frac = _vec_from_feats(feats, ENGINE["feat_cols"], mu_by_index, set(UNCOMP_BASE))
        self._neut_frac = neut_frac; self.feat_hist.append(v)
        enough_pose = np.mean(list(self.pose_present)) >= 0.5
        quality_ok = (self._coverage >= QUALITY_COVERAGE_MIN) and (self._neut_frac <= QUALITY_NEUTRALIZE_MAX)

        if len(self.feat_hist) >= 11:
            recent = np.diff(np.stack(list(self.feat_hist)[-11:], axis=0), axis=0)
            self._mov_score = float(np.mean(np.abs(recent)))
        else:
            self._mov_score = 0.0

        now = time.time()
        if self._mov_score >= ACTIVE_MOV_TH:
            self.last_active_ts = now
        if has_pose and (self._coverage >= 0.40):
            self.last_pose_ts = now
        if (self._mov_score < REST_MOV_TH) or (self._coverage < 0.30) or (not has_pose):
            if self.idle_started_ts is None:
                self.idle_started_ts = now
        else:
            self.idle_started_ts = None

        if not self.ready:
            if (WARMUP_PAD and len(self.feat_hist) >= self.prob_hist.maxlen) or (len(self.feat_hist) >= 12):
                self.ready = True
                LOG("engine | ready")
            else:
                self.status = "need_full_body" if not enough_pose else "warming"

        avg_probs = None
        if self.ready and enough_pose and quality_ok:
            tail_len = self.feat_hist.maxlen
            X_tail = np.stack(list(self.feat_hist)[-tail_len:], axis=0)
            pad = ENGINE["seq_len"] - X_tail.shape[0]
            if pad > 0:
                mu_row = ENGINE["mu"].astype(np.float32)
                X_raw = np.vstack([np.tile(mu_row, (pad, 1)), X_tail])
            else:
                X_raw = X_tail[-ENGINE["seq_len"]:]
            X = (X_raw - ENGINE["mu"]) / ENGINE["sd"]
            xb = torch.from_numpy(X[None, ...].astype(np.float32)).to(ENGINE["device"])
            with torch.no_grad():
                logits_type, _ = ENGINE["model"](xb)
                probs = torch.softmax(logits_type, dim=1).cpu().numpy()[0]

            if self._mov_score >= MOV_RECORD_THRESH:
                self.prob_hist.append(probs)
            avg_probs = np.mean(np.stack(self.prob_hist, axis=0), axis=0) if len(self.prob_hist) else probs

            idx_map = {n.lower(): i for i, n in enumerate(ENGINE["type_names"])}
            sel_idx = idx_map.get(str(self.selected).strip().lower(), None)
            if sel_idx is not None:
                self.last_label = ENGINE["type_names"][sel_idx]
                self.last_conf = float(avg_probs[sel_idx])
            else:
                self.last_label = "unknown"; self.last_conf = 0.0

            self.status = "ok"
            tip = self.coach.update(self.last_label, self.last_raw, mov=self._mov_score, coverage=self._coverage)
            if (not tip) and tips_from_phase_events and (self.squat_pd or self.jacks_pd):
                events = []
                ll = self.last_label.lower()
                if self.squat_pd and ll in ("squat", "squats"):
                    lk = float(self.last_raw.get("left_knee_angle", 180) or 180)
                    rk = float(self.last_raw.get("right_knee_angle", 180) or 180)
                    events += self.squat_pd.update(self.frame_i, min(lk, rk))
                if self.jacks_pd and ll in ("jumping jacks",):
                    la = float(self.last_raw.get("left_shoulder_abd", 0) or 0)
                    ra = float(self.last_raw.get("right_shoulder_abd", 0) or 0)
                    events += self.jacks_pd.update(self.frame_i, max(la, ra))
                pts = tips_from_phase_events(events, {
                    "knee_angle": min(float(self.last_raw.get("left_knee_angle", 180) or 180),
                                      float(self.last_raw.get("right_knee_angle", 180) or 180)),
                    "trunk_tilt": float(self.last_raw.get("trunk_tilt", 0) or 0),
                    "shoulder_abd": max(float(self.last_raw.get("left_shoulder_abd", 0) or 0),
                                        float(self.last_raw.get("right_shoulder_abd", 0) or 0)),
                }) if events else []
                if not tip and pts:
                    tip = pts[0]

            if not tip:
                tip = _simple_tip(self.last_label, self.last_raw)
            self.last_tip = tip if self.last_label != "unknown" else None

            if self.show_debug and avg_probs is not None:
                top3_idx = avg_probs.argsort()[-3:][::-1]
                top3_str = ", ".join(f"{ENGINE['type_names'][i]}={avg_probs[i]:.2f}" for i in top3_idx)
                LOG(f"top3 | {top3_str}")

            LOG(f"pred={self.last_label} conf={self.last_conf:.3f} mov={self._mov_score:.2f} "
                f"cov={self._coverage:.2f} neut={self._neut_frac:.2f} status={self.status}")

            if self.speak:
                if (not self._announced_ready) and (self.last_conf >= self.speak_gate) and (self._coverage >= 0.5):
                    self.speak_fn(f"{self.last_label} ready")
                    self._announced_ready = True
                    self._last_tip_t = now
                if self.last_tip and (now - self._last_tip_t > SPEAK_COOLDOWN) and (self._coverage >= 0.5):
                    self._speak_if_ok(self.last_tip)
                elif (now - getattr(self, "_last_tip_t", 0.0)) > (SPEAK_COOLDOWN * 2) and (self._coverage >= 0.5) and (self._mov_score > 0.4):
                    self._speak_if_ok("Keep going", ignore_conf=True)
            if self.last_tip:
                key = (self.last_label.lower(), self.last_tip.strip())
                last_t = self._agg_last_time.get(key, 0.0)
                if (now - last_t) >= AGG_DEDUP_S:
                    self.agg.add(self.last_label, self.last_tip)
                    self._agg_last_time[key] = now

            fps = 1.0 / max(1e-6, (time.time() - self._t_prev)); self._t_prev = time.time()
        else:
            self.prob_hist.clear()
            self.status = "need_full_body" if not enough_pose else ("low_quality" if not quality_ok else ("warming" if not self.ready else "idle"))
            if self.status != getattr(self, "_last_status", None):
                LOG(f"status={self.status} cov={self._coverage:.2f} neut={self._neut_frac:.2f} mov={self._mov_score:.2f}")
                self._last_status = self.status
            self.last_conf = 0.0
            avg_probs = None

        y = 22
        color_status = (0, 165, 255) if self.status in ("no_pose", "need_full_body", "low_quality", "warming", "idle") else (255, 255, 255)
        cv2.putText(img, f"status: {self.status}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2); y += 22
        conf_txt = f"{self.last_conf*100:.1f}%" if self.last_label != "unknown" and self.status == "ok" else "--"
        cv2.putText(img, f"pred: {self.selected}  conf: {conf_txt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 22
        fps = 1.0 / max(1e-6, (time.time() - getattr(self, "_t_prev", time.time()))) if fps is None else fps
        cv2.putText(img, f"fps: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 22
        if self.show_debug and 'avg_probs' in locals() and avg_probs is not None:
            top3_idx = avg_probs.argsort()[-3:][::-1]
            debug_top3 = " | ".join(f"{ENGINE['type_names'][i]}:{avg_probs[i]*100:.0f}%" for i in top3_idx)
            cv2.putText(img, f"top3: {debug_top3}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2); y += 22
        if self.show_debug:
            cv2.putText(img, f"neutralized: {int(self._neut_frac*100)}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2); y += 22
            cv2.putText(img, f"coverage: {int(self._coverage*100)}%  lower: {int(self._lower_cov*100)}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2); y += 22
            cv2.putText(img, f"motion:{self._mov_score:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 200, 180), 2); y += 22
        if self.speak and (self.agg.total > 0):
            idle_ok = (self.idle_started_ts is not None) and ((now - self.idle_started_ts) >= REST_SECS)
            nopose_ok = (now - self.last_pose_ts) >= NOPOSE_REST_SECS
            if (idle_ok or nopose_ok) and ((now - self.last_summary_ts) >= SUMMARY_COOLDOWN_S):
                self.speak_summary(k=3, min_count=2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("PoseCoachAI — Real-time Coach")

@st.cache_resource(show_spinner=False)
def get_tts(cache_buster: int = 7):
    return TTS(prefer_browser=False, rate=170, volume=1.0, beep=False)

tts = get_tts()

cols_test = st.columns([1, 1, 2, 2])
with cols_test[0]:
    if st.button("Speak test", key="speak_test_debug"):
        tts.say("This is a PoseCoach test.")
with cols_test[1]:
    beep_debug = st.toggle("Beep (debug)", value=False, key="beep_toggle_debug")
    try:
        tts.set_beep(beep_debug)
    except AttributeError:
        pass
with cols_test[2]:
    if st.button("Beep test only", key="beep_test_only"):
        try:
            tts.beep_once()
        except AttributeError:
            pass
LOG("tts_backend=server")

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    selected_ex = st.selectbox("Select exercise", options=ENGINE["type_names"], index=0, key="ex_select")
with col2:
    speak_enabled = st.toggle("Voice tips", value=True, key="voice_tips_toggle")
with col3:
    speak_gate = st.slider("Speak ≥", 0.0, 1.0, DEFAULT_SPEAK_GATE, 0.05, key="speak_gate_slider")
with col4:
    show_debug = st.toggle("Debug overlays", value=False, key="debug_overlay_toggle")

st.markdown("### Session Controls")
toolbar = st.container()
with toolbar:
    c1, c2, c3 = st.columns([2, 2, 6])
    with c1:
        end_clicked = st.button("End set ▶ Speak summary", type="primary", use_container_width=True)
    with c2:
        tips_count_placeholder = st.empty()
    with c3:
        if st.session_state.last_summary_text:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(st.session_state.last_summary_ts or time.time()))
            st.download_button(
                "Download last summary",
                data=st.session_state.last_summary_text,
                file_name=f"posecoach_set_{ts}.txt",
                mime="text/plain",
                use_container_width=True, )
        else:
            st.button("Download last summary", disabled=True, use_container_width=True)

ctx = webrtc_streamer(
    key="posecoach",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width": 640, "height": 360, "frameRate": 24}, "audio": False},
    video_processor_factory=lambda: OverlayProcessor(
        selected=selected_ex,
        speak=speak_enabled,
        speak_gate=speak_gate,
        show_debug=show_debug,
        speak_fn=tts.say,
        infer_seq_len=INFER_SEQ_LEN_DEFAULT,
        smooth_k=SMOOTH_K_DEFAULT, ),
    rtc_configuration=RTC_CONFIGURATION,)

if ctx.video_processor is not None:
    vp = ctx.video_processor
    vp.set_selected(selected_ex)
    vp.set_speak(speak_enabled, gate=speak_gate)
    vp.set_debug(show_debug)
    vp.speak_fn = tts.say

    tips_count_placeholder.metric("Tips this set", vp.tips_this_set)

    if end_clicked:
        summary = vp.speak_summary(k=3, min_count=2)
        if summary:
            (st.toast if hasattr(st, "toast") else st.info)("Speaking set summary…")
        else:
            (st.toast if hasattr(st, "toast") else st.info)("No tips collected this set.")


tts.render()


