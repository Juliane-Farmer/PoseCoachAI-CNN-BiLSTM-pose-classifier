import time
import sys
import logging
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from posecoach.inference.engine import load_engine
from overlay_processor import OverlayProcessor, DEFAULT_SPEAK_GATE
from posecoach.tts import TTS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "camera_paused" not in st.session_state:
    st.session_state.camera_paused = False
if "last_summary_text" not in st.session_state:
    st.session_state.last_summary_text = ""
if "last_summary_ts" not in st.session_state:
    st.session_state.last_summary_ts = 0.0
if "last_summary_ts_seen" not in st.session_state:
    st.session_state.last_summary_ts_seen = 0.0

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
    fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

LOGGER = _build_logger()
def LOG(msg, level="info"):
    print(msg, flush=True)
    getattr(LOGGER, level)(msg)

@st.cache_resource(show_spinner=False)
def get_engine():
    return load_engine(ROOT)

@st.cache_resource(show_spinner=False)
def get_tts(cache_buster: int = 1):
    return TTS(prefer_browser=False, rate=170, volume=1.0, beep=False)

ENGINE = get_engine()
tts = get_tts()
LOG("engine+tts ready")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="posecoach_autorefresh")
except Exception:
    pass

st.title("PoseCoachAI — Real-time Coach")

cols_test = st.columns([1, 1, 2, 2])
with cols_test[0]:
    if st.button("Speak test"):
        tts.say("This is a PoseCoach test.")
with cols_test[1]:
    beep_debug = st.toggle("Beep (debug)", value=False)
    try: tts.set_beep(beep_debug)
    except Exception: pass
with cols_test[2]:
    if st.button("Beep test only"):
        try: tts.beep_once()
        except Exception: pass

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
with col1:
    selected_ex = st.selectbox("Select exercise", options=ENGINE["type_names"], index=0)
with col2:
    speak_enabled = st.toggle("Voice tips", value=True)
with col3:
    speak_gate = st.slider("Speak ≥", 0.0, 1.0, DEFAULT_SPEAK_GATE, 0.05)
with col4:
    show_debug = st.toggle("Debug overlays", value=False)
with col5:
    st.session_state.auto_pause_on_summary = st.toggle("Auto-pause camera on summary", value=False)

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
                use_container_width=True,)
        else:
            st.button("Download last summary", disabled=True, use_container_width=True)

ctx = None
if not st.session_state.camera_paused:
    ctx = webrtc_streamer(
        key="posecoach",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": {"width": 640, "height": 360, "frameRate": 24}, "audio": False},
        video_processor_factory=lambda: OverlayProcessor(
            engine=ENGINE,
            selected=selected_ex,
            speak=speak_enabled,
            speak_gate=speak_gate,
            show_debug=show_debug,
            speak_fn=tts.say,),
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
                vp.reset_set()  
                tips_count_placeholder.metric("Tips this set", vp.tips_this_set)
                if st.session_state.auto_pause_on_summary:
                    st.session_state.camera_paused = True
                    try: ctx.stop()
                    except Exception: pass
                    st.experimental_rerun()
            else:
                (st.toast if hasattr(st, "toast") else st.info)("No tips collected this set.")
        if st.session_state.last_summary_ts > st.session_state.last_summary_ts_seen:
            st.session_state.last_summary_ts_seen = st.session_state.last_summary_ts
            vp.reset_set()
            tips_count_placeholder.metric("Tips this set", vp.tips_this_set)
            if st.session_state.auto_pause_on_summary:
                st.session_state.camera_paused = True
                try: ctx.stop()
                except Exception: pass
                st.experimental_rerun()
else:
    st.info("Camera paused. Press START to begin the next set.")
    if st.button("START camera", type="primary", use_container_width=True):
        st.session_state.camera_paused = False
        st.experimental_rerun()

tts.render()
