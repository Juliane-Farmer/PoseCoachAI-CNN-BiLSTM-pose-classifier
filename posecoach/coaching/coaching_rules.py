import numpy as np
from collections import deque

COVERAGE_MIN = 0.50     
MOTION_MIN   = 0.18    
CONSIST_WIN  = 12    
MIN_HIST     = 8       

def _valid(v): return v is not None and not (isinstance(v, float) and np.isnan(v))
def _nanmin(*xs):
    xs = [x for x in xs if _valid(x)]
    return min(xs) if xs else np.nan
def _nanmax(*xs):
    xs = [x for x in xs if _valid(x)]
    return max(xs) if xs else np.nan
def _nanmean(*xs):
    xs = [x for x in xs if _valid(x)]
    return float(np.mean(xs)) if xs else np.nan

def _stance_ratio(raw):
    if not all(k in raw for k in ("left_ankle_x","right_ankle_x","left_hip_x","right_hip_x")):
        return np.nan
    ax = raw.get("left_ankle_x"); bx = raw.get("right_ankle_x")
    hx = raw.get("left_hip_x");   kx = raw.get("right_hip_x")
    if not all(map(_valid, (ax,bx,hx,kx))): return np.nan
    hip_w  = abs(hx - kx); feet_w = abs(ax - bx)
    if hip_w <= 1e-6: return np.nan
    return feet_w / hip_w

def _arms_overhead(raw, abd_now, lift=0.02):
    head_y = _nanmin(raw.get("left_ear_y"), raw.get("right_ear_y"), raw.get("nose_y"))
    lw = raw.get("left_wrist_y", np.nan)
    rw = raw.get("right_wrist_y", np.nan)
    if _valid(head_y) and _valid(lw) and _valid(rw):
        return (lw < head_y - lift) and (rw < head_y - lift)
    return abd_now >= 110

class FormCoach:
    """
    Real-time form cues tuned to the signals available in PoseCoachAI.
    Uses short history to (a) avoid spam, (b) gate praise on consistency, (c) keep tips actionable.
    """
    SQUAT_TILT_MAX = 25         
    SQUAT_DEPTH_GOOD = 95       
    STANCE_MIN_XR = 0.8        
    STANCE_MAX_XR = 1.6

    PUSH_TILT_MAX  = 18        
    PUSH_HIP_MIN   = 165      
    PUSH_ELBOW_MIN = 95         
    PUSH_LOCKOUT   = 165       

    PULL_TILT_MAX  = 15
    PULL_TOP_ELBOW = 75      
    PULL_HANG_ELBOW= 160     

    JACKS_MIN_XR   = 1.2       
    JACKS_RHYTHM   = 0.60      

    RUS_TILT_MAX   = 20       
    RUS_MIN_MOV    = 0.35   

    def __init__(self):
        self.knee_hist  = deque(maxlen=24)
        self.elbow_hist = deque(maxlen=24)
        self.hip_hist   = deque(maxlen=24)
        self.tilt_hist  = deque(maxlen=24)
        self.abd_hist   = deque(maxlen=24)
        self._last_ex   = None
        self._praise_cool = 0  

    def _trend(self, hist):
        if len(hist) < 3: return 0.0
        x = np.asarray(hist, dtype=float)
        if np.any(np.isnan(x[-3:])): return 0.0
        return 0.5 * ((x[-1]-x[-2]) + (x[-2]-x[-3]))

    def update(self, exercise: str, raw: dict, mov: float = 0.0, coverage: float = 1.0):
        ex = (exercise or "").lower().strip()
        if ex != self._last_ex:
            self.knee_hist.clear(); self.elbow_hist.clear()
            self.hip_hist.clear();  self.tilt_hist.clear(); self.abd_hist.clear()
            self._praise_cool = 0
            self._last_ex = ex
        knee_min  = _nanmin(raw.get("left_knee_angle"),  raw.get("right_knee_angle"))
        elbow_min = _nanmin(raw.get("left_elbow_angle"), raw.get("right_elbow_angle"))
        hip_min   = _nanmin(raw.get("left_hip_angle"),   raw.get("right_hip_angle"))
        tilt      = raw.get("trunk_tilt", np.nan)
        abd_min   = _nanmin(raw.get("left_shoulder_abd"), raw.get("right_shoulder_abd"))
        xr        = _stance_ratio(raw)

        if _valid(knee_min):  self.knee_hist.append(float(knee_min))
        if _valid(elbow_min): self.elbow_hist.append(float(elbow_min))
        if _valid(hip_min):   self.hip_hist.append(float(hip_min))
        if _valid(tilt):      self.tilt_hist.append(float(tilt))
        if _valid(abd_min):   self.abd_hist.append(float(abd_min))
        if coverage < COVERAGE_MIN or mov < MOTION_MIN:
            self._praise_cool = max(0, self._praise_cool - 1)
            return None

        knee_trend  = self._trend(self.knee_hist)
        elbow_trend = self._trend(self.elbow_hist)
        tilt_now    = float(tilt) if _valid(tilt) else 0.0
        abd_now     = float(abd_min) if _valid(abd_min) else 0.0

        if ex in ("squat", "squats"):
            if _valid(xr) and not (self.STANCE_MIN_XR <= xr <= self.STANCE_MAX_XR):
                return "Set feet about shoulder-width"
            if tilt_now > self.SQUAT_TILT_MAX:
                return "Keep chest up—neutral spine"
            if len(self.knee_hist) >= MIN_HIST:
                if np.nanmin(list(self.knee_hist)[-CONSIST_WIN:]) > self.SQUAT_DEPTH_GOOD:
                    return "Go deeper—hips to at least parallel"
                if knee_trend < -1.0 and (_valid(knee_min) and knee_min > 110):
                    return "Control the descent"
                if knee_trend > +1.0 and (_valid(knee_min) and knee_min < 95):
                    return "Drive up"
            if (len(self.knee_hist) >= CONSIST_WIN
                and np.nanmin(list(self.knee_hist)[-CONSIST_WIN:]) <= self.SQUAT_DEPTH_GOOD
                and tilt_now <= self.SQUAT_TILT_MAX):
                if self._praise_cool == 0:
                    self._praise_cool = 20
                    return "Nice squat!"
            self._praise_cool = max(0, self._praise_cool - 1)
            return None

        if ex in ("jumping jacks","jumping-jacks","jumpingjack"):
            if not _arms_overhead(raw, abd_now):
                return "Touch hands fully overhead"
            if _valid(xr) and xr < self.JACKS_MIN_XR:
                return "Jump wider with your feet"
            if mov < self.JACKS_RHYTHM:
                return "Keep a steady rhythm"
            if mov >= self.JACKS_RHYTHM and _arms_overhead(raw, abd_now) and (_valid(xr) and xr >= self.JACKS_MIN_XR):
                if self._praise_cool == 0:
                    self._praise_cool = 18
                    return "Great rhythm"
            self._praise_cool = max(0, self._praise_cool - 1)
            return None

        if ex in ("russian twists","russian twist"):
            if tilt_now > self.RUS_TILT_MAX:
                return "Sit tall—avoid rounding the back"
            if mov < self.RUS_MIN_MOV:
                return "Rotate shoulders more side-to-side"
            if mov >= self.RUS_MIN_MOV and tilt_now <= self.RUS_TILT_MAX:
                if self._praise_cool == 0:
                    self._praise_cool = 20
                    return "Nice controlled twists"
            self._praise_cool = max(0, self._praise_cool - 1)
            return None

        if ex in ("pushup","push-ups","push ups"):
            if tilt_now > self.PUSH_TILT_MAX:
                return "Keep body in one line—brace core"
            if _valid(hip_min) and hip_min < self.PUSH_HIP_MIN:
                return "Lift hips—avoid sagging"
            if len(self.elbow_hist) >= MIN_HIST and np.nanmin(self.elbow_hist) > self.PUSH_ELBOW_MIN:
                return "Go deeper at the bottom"
            if elbow_trend < -1.2 and (_valid(elbow_min) and elbow_min > 110):
                return "Slow the descent"
            if elbow_trend > +1.2 and (_valid(elbow_min) and elbow_min < 90):
                return "Press up strong"
            if (len(self.elbow_hist) >= CONSIST_WIN
                and np.nanmax(list(self.elbow_hist)[-CONSIST_WIN:]) >= self.PUSH_LOCKOUT
                and tilt_now <= self.PUSH_TILT_MAX
                and (_valid(hip_min) and hip_min >= self.PUSH_HIP_MIN)):
                if self._praise_cool == 0:
                    self._praise_cool = 22
                    return "Nice clean push-ups"
            self._praise_cool = max(0, self._praise_cool - 1)
            return None

        if ex in ("pull ups","pull-ups","pullup"):
            if tilt_now > self.PULL_TILT_MAX:
                return "Keep your torso stable"
            if len(self.elbow_hist) >= MIN_HIST and np.nanmax(self.elbow_hist) < self.PULL_HANG_ELBOW:
                return "Reach a fuller hang at the bottom"
            if len(self.elbow_hist) >= MIN_HIST and np.nanmin(self.elbow_hist) > self.PULL_TOP_ELBOW:
                return "Pull higher—aim chest to bar"
            if (len(self.elbow_hist) >= CONSIST_WIN
                and np.nanmin(list(self.elbow_hist)[-CONSIST_WIN:]) <= self.PULL_TOP_ELBOW
                and np.nanmax(list(self.elbow_hist)[-CONSIST_WIN:]) >= self.PULL_HANG_ELBOW
                and tilt_now <= self.PULL_TILT_MAX):
                if self._praise_cool == 0:
                    self._praise_cool = 24
                    return "Great pull up."
            self._praise_cool = max(0, self._praise_cool - 1)
            return None

        self._praise_cool = max(0, self._praise_cool - 1)
        return None
