import numpy as np
from collections import deque

def _nanmin(*vals, default=np.nan):
    xs = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return min(xs) if xs else default

class FormCoach:
    def __init__(self):
        self.knee_hist = deque(maxlen=24)
        self.tilt_hist = deque(maxlen=24)
        self.abd_hist  = deque(maxlen=24)
        self._last_ex = None

    def _trend(self, hist):
        if len(hist) < 3:
            return 0.0
        x = np.asarray(hist, dtype=float)
        if np.any(np.isnan(x[-3:])):
            return 0.0
        return 0.5 * ((x[-1] - x[-2]) + (x[-2] - x[-3]))

    def update(self, exercise: str, raw: dict, mov: float = 0.0, coverage: float = 1.0):
        ex = (exercise or "").lower()
        if ex != self._last_ex:
            self.knee_hist.clear(); self.tilt_hist.clear(); self.abd_hist.clear()
            self._last_ex = ex

        knee_l = raw.get("left_knee_angle", np.nan)
        knee_r = raw.get("right_knee_angle", np.nan)
        knee_min = _nanmin(knee_l, knee_r)
        tilt = raw.get("trunk_tilt", np.nan)
        abd_l = raw.get("left_shoulder_abd", np.nan)
        abd_r = raw.get("right_shoulder_abd", np.nan)
        abd_min = _nanmin(abd_l, abd_r)

        if not np.isnan(knee_min): self.knee_hist.append(float(knee_min))
        if not np.isnan(tilt):     self.tilt_hist.append(float(tilt))
        if not np.isnan(abd_min):  self.abd_hist.append(float(abd_min))

        knee_trend = self._trend(self.knee_hist)
        tilt_now = 0.0 if (isinstance(tilt, float) and np.isnan(tilt)) else float(tilt)
        abd_now = 0.0 if (isinstance(abd_min, float) and np.isnan(abd_min)) else float(abd_min)

        if coverage < 0.45 or mov < 0.20:
            return None

        if ex in ("squat", "squats"):
            if tilt_now > 28: return "Keep chest up"
            if len(self.knee_hist) >= 10 and np.nanmin(self.knee_hist) > 120: return "Bend knees more"
            if knee_trend < -1.0 and (np.isnan(knee_min) or knee_min > 110): return "Go deeper"
            if knee_trend > +1.0 and (not np.isnan(knee_min) and knee_min < 95): return "Drive up"
            return None

        if ex in ("jumping jacks", "jumping-jacks", "jumpingjack"):
            head_y = np.nanmin([
                raw.get("left_ear_y", np.nan),
                raw.get("right_ear_y", np.nan),
                raw.get("nose_y", np.nan) ])
            lw = raw.get("left_wrist_y", np.nan)
            rw = raw.get("right_wrist_y", np.nan)
            arms_up = False
            if not np.isnan(head_y) and not np.isnan(lw) and not np.isnan(rw):
                arms_up = (lw < head_y - 0.02) and (rw < head_y - 0.02)
            else:
                arms_up = abd_now >= 110
            if not arms_up: return "Hands overhead"
            if mov < 0.6: return "Keep the rhythm"
            return None

        if ex in ("russian twists", "russian twist"):
            if tilt_now > 25: return "Sit tall"
            return "Rotate shoulders more"

        if ex in ("pushup", "push-ups", "push ups"):
            if tilt_now > 18: return "Body in one line"
            return None
        return None
