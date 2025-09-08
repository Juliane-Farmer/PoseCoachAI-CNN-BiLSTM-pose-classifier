import numpy as np
from collections import deque

SQUAT_TILT_MAX_DEG = 25       
SQUAT_GOOD_KNEE_MIN = 95      
STANCE_MIN_XR = 0.8         
STANCE_MAX_XR = 1.6

JACKS_ABD_ARMS_UP = 110       
JACKS_MIN_STANCE_XR = 1.2     
RHYTHM_MIN_MOV = 0.60

PUSHUP_TILT_MAX_DEG = 18      
PUSHUP_ELBOW_MIN = 95       
PUSHUP_HIP_MIN = 165          

PULLUP_TILT_MAX_DEG = 15     
PULLUP_ELBOW_TOP = 75         

RUSSIAN_TILT_MAX_DEG = 20     
RUSSIAN_MIN_MOV = 0.35        

COVERAGE_MIN = 0.45          
MOTION_MIN = 0.20

def _valid_num(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))

def _nanmin(*vals, default=np.nan):
    xs = [v for v in vals if _valid_num(v)]
    return min(xs) if xs else default

def _nanmean(*vals, default=np.nan):
    xs = [v for v in vals if _valid_num(v)]
    return float(np.mean(xs)) if xs else default

class FormCoach:
  
    def __init__(self):
        self.knee_hist = deque(maxlen=24)
        self.elbow_hist = deque(maxlen=24)
        self.hip_hist   = deque(maxlen=24)
        self.tilt_hist  = deque(maxlen=24)
        self.abd_hist   = deque(maxlen=24)
        self._last_ex = None

    def _trend(self, hist):
        if len(hist) < 3: return 0.0
        x = np.asarray(hist, dtype=float)
        if np.any(np.isnan(x[-3:])): return 0.0
        return 0.5 * ((x[-1] - x[-2]) + (x[-2] - x[-3]))

    def _stance_ratio(self, raw):
        """
        Feet width relative to hip width using X coordinates if available.
        XR = |ankle_L.x - ankle_R.x| / |hip_L.x - hip_R.x|
        Returns np.nan if not enough landmarks.
        """
        if not all(k in raw for k in ("left_ankle_x","right_ankle_x","left_hip_x","right_hip_x")):
            return np.nan
        ax = raw.get("left_ankle_x", np.nan)
        bx = raw.get("right_ankle_x", np.nan)
        hx = raw.get("left_hip_x", np.nan)
        kx = raw.get("right_hip_x", np.nan)
        if not all(map(_valid_num, (ax,bx,hx,kx))): return np.nan
        hip_w = abs(hx - kx)
        feet_w = abs(ax - bx)
        if hip_w <= 1e-6: return np.nan
        return feet_w / hip_w

    def _arms_overhead(self, raw, abd_now):
        head_y = _nanmin(raw.get("left_ear_y"), raw.get("right_ear_y"), raw.get("nose_y"))
        lw = raw.get("left_wrist_y", np.nan)
        rw = raw.get("right_wrist_y", np.nan)
        if _valid_num(head_y) and _valid_num(lw) and _valid_num(rw):
            return (lw < head_y - 0.02) and (rw < head_y - 0.02)
        return abd_now >= JACKS_ABD_ARMS_UP

    def update(self, exercise: str, raw: dict, mov: float = 0.0, coverage: float = 1.0):
        ex = (exercise or "").lower().strip()
        if ex != self._last_ex:
            self.knee_hist.clear(); self.elbow_hist.clear()
            self.hip_hist.clear();  self.tilt_hist.clear(); self.abd_hist.clear()
            self._last_ex = ex

        knee_l = raw.get("left_knee_angle", np.nan)
        knee_r = raw.get("right_knee_angle", np.nan)
        knee_min = _nanmin(knee_l, knee_r)

        elbow_l = raw.get("left_elbow_angle", np.nan)
        elbow_r = raw.get("right_elbow_angle", np.nan)
        elbow_min = _nanmin(elbow_l, elbow_r)

        hip_l = raw.get("left_hip_angle", np.nan)
        hip_r = raw.get("right_hip_angle", np.nan)
        hip_min = _nanmin(hip_l, hip_r)

        tilt = raw.get("trunk_tilt", np.nan)
        abd_l = raw.get("left_shoulder_abd", np.nan)
        abd_r = raw.get("right_shoulder_abd", np.nan)
        abd_min = _nanmin(abd_l, abd_r)

        if _valid_num(knee_min):  self.knee_hist.append(float(knee_min))
        if _valid_num(elbow_min): self.elbow_hist.append(float(elbow_min))
        if _valid_num(hip_min):   self.hip_hist.append(float(hip_min))
        if _valid_num(tilt):      self.tilt_hist.append(float(tilt))
        if _valid_num(abd_min):   self.abd_hist.append(float(abd_min))

        knee_trend  = self._trend(self.knee_hist)
        elbow_trend = self._trend(self.elbow_hist)
        tilt_now = float(tilt) if _valid_num(tilt) else 0.0
        abd_now  = float(abd_min) if _valid_num(abd_min) else 0.0
        xr = self._stance_ratio(raw)  
        if coverage < COVERAGE_MIN or mov < MOTION_MIN:
            return None

        if ex in ("squat", "squats"):
            if _valid_num(xr) and not (STANCE_MIN_XR <= xr <= STANCE_MAX_XR):
                return "Set feet ~shoulder-width apart"
            if tilt_now > SQUAT_TILT_MAX_DEG:
                return "Keep chest up (neutral spine)"
            if len(self.knee_hist) >= 8:
                if np.nanmin(self.knee_hist) > SQUAT_GOOD_KNEE_MIN:
                    return "Go deeper (hips to at least parallel)"
            if knee_trend < -1.0 and (_valid_num(knee_min) and knee_min > 110):
                return "Control the descent"
            if knee_trend > +1.0 and (_valid_num(knee_min) and knee_min < 95):
                return "Drive up"
            return None

        if ex in ("jumping jacks", "jumping-jacks", "jumpingjack"):
            if not self._arms_overhead(raw, abd_now):
                return "Get hands fully overhead each rep"
            if _valid_num(xr) and xr < JACKS_MIN_STANCE_XR:
                return "Jump wider with your feet"
            if mov < RHYTHM_MIN_MOV:
                return "Keep a steady rhythm"
            return None

        if ex in ("russian twists", "russian twist"):
            if tilt_now > RUSSIAN_TILT_MAX_DEG:
                return "Sit tall (avoid rounding the back)"
            if mov < RUSSIAN_MIN_MOV:
                return "Rotate shoulders more side-to-side"
            return None

        if ex in ("pushup", "push-ups", "push ups"):
            if tilt_now > PUSHUP_TILT_MAX_DEG:
                return "Keep body in one line (brace core)"
            if _valid_num(hip_min) and hip_min < PUSHUP_HIP_MIN:
                return "Lower hips—avoid piking"
            if len(self.elbow_hist) >= 8 and np.nanmin(self.elbow_hist) > PUSHUP_ELBOW_MIN:
                return "Go deeper—bend elbows more at the bottom"
            if elbow_trend < -1.2 and (_valid_num(elbow_min) and elbow_min > 110):
                return "Control the descent"
            if elbow_trend > +1.2 and (_valid_num(elbow_min) and elbow_min < 90):
                return "Press up strong"
            return None

        if ex in ("pull ups", "pull-ups", "pullup"):
            if tilt_now > PULLUP_TILT_MAX_DEG:
                return "Keep your body in one line (reduce swing)"
            if len(self.elbow_hist) >= 6 and np.nanmin(self.elbow_hist) > PULLUP_ELBOW_TOP:
                return "Pull higher—aim chest toward the bar"
            if mov < 0.35:
                return "Pull smoothly to the top"
            return None

        return None
