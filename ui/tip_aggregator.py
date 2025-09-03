import time
from collections import defaultdict

class TipAggregator:
    """Collects tips during a set and generates a short spoken summary."""
    def __init__(self):
        self.counts = defaultdict(int)
        self.first_ts = {}
        self.last_ts = {}
        self.total = 0

    def add(self, exercise: str, tip: str):
        if not tip: return
        key = (exercise.lower().strip(), tip.strip())
        now = time.time()
        self.counts[key] += 1
        self.last_ts[key] = now
        self.first_ts.setdefault(key, now)
        self.total += 1

    def summarize(self, k=2, min_count=2):
        if not self.counts:
            return None
        items = sorted(
            self.counts.items(),
            key=lambda kv: (kv[1], self.last_ts[kv[0]]),
            reverse=True )
        ranked = [(ex_tip[1], cnt) for ex_tip, cnt in items]
        tips = [t for t, c in ranked if c >= min_count]
        if not tips:
            tips = [t for t, _ in ranked[:k]]
        tips = tips[:k]
        if not tips: return None
        return "Set summary: " + ("; ".join(tips) + ".")

    def clear(self):
        self.counts.clear(); self.first_ts.clear(); self.last_ts.clear(); self.total = 0


