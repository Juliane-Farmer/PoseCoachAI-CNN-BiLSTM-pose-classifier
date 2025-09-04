from collections import defaultdict

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

def compose_humane_summary(counts: dict, last_ts: dict, k=2, min_count=2) -> str:
    """counts: { (exercise, tip): count } ; last_ts: { (exercise, tip): ts }"""
    if not counts:
        return "Nice work—no major corrections this set."
    ex_counts = defaultdict(int)
    for (ex, tip), c in counts.items():
        ex_counts[ex] += c
    exercise = max(ex_counts.items(), key=lambda kv: kv[1])[0]
    items = sorted(counts.items(), key=lambda kv: (kv[1], last_ts[kv[0]]), reverse=True)
    phrases = []
    for (ex, tip), cnt in items:
        if ex != exercise: continue
        if cnt < min_count and len(phrases) >= k: continue
        phrases.append(REPHRASE.get(tip, tip.lower()))
        if len(phrases) >= k: break
    if not phrases and items:
        phrases = [REPHRASE.get(items[0][0][1], items[0][0][1].lower())]
    friendly_ex = exercise.capitalize()
    return f"Here's some feedback from this set of {friendly_ex}: " + "; ".join(phrases) + "."
