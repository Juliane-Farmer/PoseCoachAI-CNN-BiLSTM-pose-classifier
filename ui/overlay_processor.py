import time
import json
import logging
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import torch
import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase

from posecoach.coaching.tip_aggregator import TipAggregator
from posecoach.coaching.coaching_rules import FormCoach
try:
    from posecoach.coaching.coaching_rules import tips_from_phase_events
except Exception:
    tips_from_phase_events = None
try:
    from posecoach.coaching.phase_detectors import SquatPhaseDetector, JacksPhaseDetector
except Exception:
    SquatPhaseDetector = JacksPhaseDetector = None

from posecoach.vision.pose_utils import (
    ANGLE_SPECS, REQUIRED_JOINTS, VIS_THRESH,
    safe_joint_angle, trunk_tilt_deg, get_xyzv)
from posecoach.coaching.summary import compose_humane_summary

LOGGER = logging.getLogger("posecoach")

SMOOTH_K_DEFAULT = 6
WARMUP_PAD = True
DEFAULT_SPEAK_GATE = 0.30
QUALITY_NEUTRALIZE_MAX = 0.35
QUALITY_COVERAGE_MIN = 0.60
MOV_RECORD_THRESH = 0.25
SPEAK_COOLDOWN = 2.0

REST_SECS = 15.0
NOPOSE_REST_SECS = 15.0
ACTIVE_MOV_TH = 0.18
REST_MOV_TH = 0.12
SUMMARY_COOLDOWN_S = 10.0
AGG_DEDUP_S = 1.0

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs" / "session_logs" / "last_summary.json"

def _base_of(n: str) -> str:
    if n.endswith("_diff"): return n[:-5]
    if n.endswith("_ma5"): return n[:-4]
    return n

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

class OnlineFeatureBuilder:
    def __init__(self, base_names, ma_window=5):
        self.base = list(base_names)
        self.prev = defaultdict(lambda: None)
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

class OverlayProcessor(VideoProcessorBase):
    def __init__(self, engine, selected, speak=False, speak_gate=DEFAULT_SPEAK_GATE, show_debug=False,
                 speak_fn=None, infer_seq_len=None, smooth_k=SMOOTH_K_DEFAULT):
        self.engine = engine
        self.selected = selected
        self.show_debug = show_debug
        self.pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.prob_hist = deque(maxlen=int(smooth_k))
        seq = max(40, min(engine["seq_len"], 64)) if infer_seq_len is None else int(infer_seq_len)
        self.feat_hist = deque(maxlen=seq)
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
        self._last_summary_payload = None
        self._start_ts = time.time()
        self._warmup_announced = False
        self._ready_announced = False
        self._feat_cols = self.engine["feat_cols"]
        self._mu = self.engine["mu"]
        self._sd = self.engine["sd"]
        self._mu_by_index = {i: float(self._mu[i]) for i,_ in enumerate(self._feat_cols)}
        required_base = { _base_of(n) for n in self._feat_cols } | {"trunk_tilt"}
        computable = set(ANGLE_SPECS.keys()) | {"trunk_tilt"}
        self._uncomp_base = sorted(b for b in required_base if b not in computable)
        self._required_base = sorted(required_base)

    @property
    def tips_this_set(self):
        return self.agg.total

    def reset_set(self):
        self.agg.clear(); self._agg_last_time.clear()
        self.last_active_ts = time.time()
        self.last_pose_ts = time.time()
        self.idle_started_ts = None
        LOGGER.info("set | reset")

    def set_selected(self, s): self.selected = s
    def set_speak(self, enabled, gate=DEFAULT_SPEAK_GATE): self.speak = enabled; self.speak_gate = gate
    def set_debug(self, val): self.show_debug = bool(val)

    def _compose_summary(self, k=2, min_count=2):
        return compose_humane_summary(self.agg.counts, self.agg.last_ts, k=k, min_count=min_count)

    def speak_summary(self, k=2, min_count=2):
        summary = self._compose_summary(k=k, min_count=min_count)
        if summary:
            ts = time.time()
            self.last_summary_ts = ts
            self._last_summary_payload = {"text": summary, "ts": ts}
            try:
                SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                    json.dump({"text": summary, "ts": ts}, f)
            except Exception as e:
                LOGGER.warning(f"summary_persist_fail: {e}")
            try:
                if self.speak:
                    self.speak_fn(summary)
            except Exception:
                pass
            self.agg.clear(); self._agg_last_time.clear()
            LOGGER.info(f"speak_summary | {summary}")
        return summary

    def consume_summary_payload(self):
        p, self._last_summary_payload = self._last_summary_payload, None
        return p

    def _record_tip_heard(self, tip: str):
        key = (self.last_label.lower(), (tip or "").strip())
        last_t = self._agg_last_time.get(key, 0.0)
        if (time.time() - last_t) >= AGG_DEDUP_S:
            self.agg.add(self.last_label, tip)
            self._agg_last_time[key] = time.time()

    def _speak_if_ok(self, tip, ignore_conf=False):
        reasons = []
        if not self.speak: reasons.append("speak=False")
        if self._coverage < 0.50: reasons.append(f"cov={self._coverage:.2f}<0.50")
        if (not ignore_conf) and (self.last_conf < self.speak_gate): reasons.append(f"conf={self.last_conf:.2f}<gate={self.speak_gate:.2f}")
        if not tip: reasons.append("no-tip")
        if reasons:
            LOGGER.info("speak_skip | " + ", ".join(reasons)); return
        self.speak_fn(tip)
        self._last_tip = tip
        self._last_tip_t = time.time()
        if tip not in ("Keep going", f"{self.last_label} ready"):
            self._record_tip_heard(tip)
        LOGGER.info(f"speak | {tip}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_i += 1
        fps = None
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        has_pose = bool(res.pose_landmarks); self.pose_present.append(has_pose)
        lm = res.pose_landmarks.landmark if has_pose else None

        all_joints = sorted(set(sum([list(v) for v in REQUIRED_JOINTS.values()], [])))
        self._coverage = (np.mean([(get_xyzv(lm, n)[1] >= VIS_THRESH) for n in all_joints]) if has_pose else 0.0)
        self._lower_cov = (np.mean([(get_xyzv(lm, n)[1] >= VIS_THRESH) for n in REQUIRED_JOINTS["squat"]]) if has_pose else 0.0)
        feats_raw = {}
        if has_pose:
            for base in self._required_base:
                if base == "trunk_tilt":
                    feats_raw["trunk_tilt"] = trunk_tilt_deg(lm)
                elif base in ANGLE_SPECS:
                    A, B, C = ANGLE_SPECS[base]; feats_raw[base] = safe_joint_angle(lm, A, B, C)
            for NM in ("LEFT_ANKLE","RIGHT_ANKLE","LEFT_HIP","RIGHT_HIP"):
                pt, vis = get_xyzv(lm, NM)
                nm = NM.lower()
                feats_raw[f"{nm}_x"] = float(pt[0]) if vis >= VIS_THRESH else np.nan
            for NM in ("LEFT_WRIST","RIGHT_WRIST"):
                pt, vis = get_xyzv(lm, NM)
                nm = NM.lower()
                feats_raw[f"{nm}_y"] = float(pt[1]) if vis >= VIS_THRESH else np.nan
        for b in self._required_base:
            feats_raw.setdefault(b, np.nan)
        if self._builder is None:
            self._builder = OnlineFeatureBuilder(self._required_base, ma_window=5)
        feats = self._builder.push(feats_raw)
        self.last_raw = feats_raw.copy()

        v, neut_frac = _vec_from_feats(feats, self._feat_cols, self._mu_by_index, set(self._uncomp_base))
        self._neut_frac = neut_frac; self.feat_hist.append(v)

        enough_pose = np.mean(list(self.pose_present)) >= 0.5
        quality_ok = (self._coverage >= QUALITY_COVERAGE_MIN) and (self._neut_frac <= QUALITY_NEUTRALIZE_MAX)

        if len(self.feat_hist) >= 11:
            recent = np.diff(np.stack(list(self.feat_hist)[-11:], axis=0), axis=0)
            self._mov_score = float(np.mean(np.abs(recent)))
        else:
            self._mov_score = 0.0
        now = time.time()

        if (not self._warmup_announced) and self.speak and (now - self._start_ts) > 1.0:
            self._warmup_announced = True
            self.speak_fn("Start with a brief warm-up set. Coaching will begin shortly.")
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
                LOGGER.info("engine | ready")
            else:
                self.status = "need_full_body" if not enough_pose else "warming"
        if self.speak and self.ready and (not self._ready_announced) and enough_pose and (self._coverage >= 0.50):
            self._ready_announced = True
            self.speak_fn("PoseCoach is ready.")

        avg_probs = None
        if self.ready and enough_pose and quality_ok:
            tail_len = self.feat_hist.maxlen
            X_tail = np.stack(list(self.feat_hist)[-tail_len:], axis=0)
            pad = self.engine["seq_len"] - X_tail.shape[0]
            if pad > 0:
                mu_row = self._mu.astype(np.float32)
                X_raw = np.vstack([np.tile(mu_row, (pad, 1)), X_tail])
            else:
                X_raw = X_tail[-self.engine["seq_len"]:]
            X = (X_raw - self._mu) / self._sd
            xb = torch.from_numpy(X[None, ...].astype(np.float32)).to(self.engine["device"])
            with torch.no_grad():
                logits_type, _ = self.engine["model"](xb)
                probs = torch.softmax(logits_type, dim=1).cpu().numpy()[0]

            if self._mov_score >= MOV_RECORD_THRESH:
                self.prob_hist.append(probs)
            avg_probs = np.mean(np.stack(self.prob_hist, axis=0), axis=0) if len(self.prob_hist) else probs

            idx_map = {n.lower(): i for i, n in enumerate(self.engine["type_names"]) }
            sel_idx = idx_map.get(str(self.selected).strip().lower(), None)
            if sel_idx is not None:
                self.last_label = self.engine["type_names"][sel_idx]
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
                if self.jacks_pd and ll in ("jumping jacks","jumping-jacks","jumpingjack"):
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
                tilt = float(self.last_raw.get("trunk_tilt", 0.0) or 0.0)
                if self.last_label.lower() in ("squat","squats"):
                    lk = float(self.last_raw.get("left_knee_angle", 180) or 180)
                    rk = float(self.last_raw.get("right_knee_angle", 180) or 180)
                    if min(lk, rk) > 160: tip = "Bend knees more"
                    elif tilt > 25: tip = "Keep chest up"
                elif self.last_label.lower() == "jumping jacks":
                    la = float(self.last_raw.get("left_shoulder_abd", 90) or 90)
                    ra = float(self.last_raw.get("right_shoulder_abd", 90) or 90)
                    if min(la, ra) < 60: tip = "Raise arms higher"

            self._last_tip = tip if self.last_label != "unknown" else None
            if self._last_tip and (self._coverage >= 0.5) and self._last_tip not in ("Keep going", f"{self.last_label} ready"):
                self._record_tip_heard(self._last_tip)
            if self.speak:
                if (not self._announced_ready) and (self.last_conf >= self.speak_gate) and (self._coverage >= 0.5):
                    self._announced_ready = True
                    self._last_tip_t = now
                    self.speak_fn(f"{self.last_label} ready")
                if self._last_tip and (now - self._last_tip_t > SPEAK_COOLDOWN) and (self._coverage >= 0.5):
                    self._speak_if_ok(self._last_tip)
                elif (now - getattr(self, "_last_tip_t", 0.0)) > (SPEAK_COOLDOWN * 2) and (self._coverage >= 0.5) and (self._mov_score > 0.4):
                    self._speak_if_ok("Keep going", ignore_conf=True)
            fps = 1.0 / max(1e-6, (time.time() - self._t_prev)); self._t_prev = time.time()
        else:
            self.prob_hist.clear()
            self.status = "need_full_body" if not enough_pose else ("low_quality" if not quality_ok else ("warming" if not self.ready else "idle"))
            if self.status != getattr(self, "_last_status", None):
                LOGGER.info(f"status={self.status} cov={self._coverage:.2f} neut={self._neut_frac:.2f} mov={self._mov_score:.2f}")
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
        if self.show_debug and avg_probs is not None:
            top3_idx = avg_probs.argsort()[-3:][::-1]
            debug_top3 = " | ".join(f"{self.engine['type_names'][i]}:{avg_probs[i]*100:.0f}%" for i in top3_idx)
            cv2.putText(img, f"top3: {debug_top3}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2); y += 22
        if self.show_debug:
            cv2.putText(img, f"neutralized: {int(self._neut_frac*100)}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2); y += 22
            cv2.putText(img, f"coverage: {int(self._coverage*100)}%  lower: {int(self._lower_cov*100)}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2); y += 22
        try:
            h, w = img.shape[:2]
            cv2.putText(img, f"tips: {self.agg.total}", (w - 160, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        except Exception:
            pass
        if self.agg.total > 0:
            idle_ok = (self.idle_started_ts is not None) and ((now - self.idle_started_ts) >= REST_SECS)
            nopose_ok = (now - self.last_pose_ts) >= NOPOSE_REST_SECS
            if (idle_ok or nopose_ok) and ((now - self.last_summary_ts) >= SUMMARY_COOLDOWN_S):
                self.speak_summary(k=3, min_count=2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
