import time
import json
import sys
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from posecoach.inference.engine import load_engine
from overlay_processor import OverlayProcessor, DEFAULT_SPEAK_GATE
from posecoach.tts import TTS

st.set_page_config(page_title="PoseCoachAI — Real-time Coach", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUMMARY_PATH = Path(__file__).resolve().parents[1] / "outputs" / "session_logs" / "last_summary.json"


def _read_summary_file():
    try:
        if SUMMARY_PATH.exists():
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
    except Exception:
        pass
    return None

ss = st.session_state
ss.setdefault("camera_paused", False)
ss.setdefault("last_summary_text", "")
ss.setdefault("last_summary_ts", 0.0)
ss.setdefault("auto_pause_on_summary", True)  

def rerun():
    try: st.rerun()
    except Exception: st.experimental_rerun()

@st.cache_resource(show_spinner=False)
def get_engine():
    return load_engine(ROOT)

@st.cache_resource(show_spinner=False)
def get_tts(cache_buster: int = 1):
    return TTS(prefer_browser=False, rate=170, volume=1.0, beep=False)

ENGINE = get_engine()
tts = get_tts()

_tick_enabled = False
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2000, key="posecoach_tick")  
    _tick_enabled = True
except Exception:
    pass

if not _tick_enabled:
    st.markdown(
        """
        <script>
        // Minimal rerun loop when `streamlit-autorefresh` is not installed.
        if (!window._posecoach_tick) {
          window._posecoach_tick = setInterval(function() {
            window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:rerun" }, "*");
          }, 2000);
        }
        </script>
        """,
        unsafe_allow_html=True, )

st.title("PoseCoachAI — Real-time Coach")

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    selected_ex = st.selectbox("Select exercise", options=ENGINE["type_names"], index=0)
with c2:
    speak_enabled = st.toggle("Voice tips", value=True)
with c3:
    speak_gate = st.slider("Speak ≥", 0.0, 1.0, DEFAULT_SPEAK_GATE, 0.05)
with st.sidebar:
    st.subheader("Debug & Utilities")
    if st.button("Speak test"):
        tts.say("This is a PoseCoach test.")
    beep_debug = st.toggle("Beep (debug)", value=False)
    try: tts.set_beep(beep_debug)
    except Exception: pass
    if st.button("Beep test only"):
        try: tts.beep_once()
        except Exception: pass
    show_debug = st.toggle("Debug overlays", value=False)
    st.divider()
    ss.auto_pause_on_summary = st.toggle("Auto-pause camera on summary", value=ss.auto_pause_on_summary)

st.markdown("### Session Controls")
t1, t2, t3 = st.columns([2, 2, 6])
with t1:
    end_clicked = st.button("End set/Speak summary", type="primary", use_container_width=True)
with t2:
    tips_count_placeholder = st.empty()
with t3:
    if ss.last_summary_text:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(ss.last_summary_ts or time.time()))
        st.download_button(
            "Download last summary",
            data=ss.last_summary_text,
            file_name=f"posecoach_set_{ts}.txt",
            mime="text/plain",
            use_container_width=True,)
    else:
        st.button("Download last summary", disabled=True, use_container_width=True)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
ctx = None

if not ss.camera_paused:
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
            speak_fn=tts.say, ),
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,  )

    if ctx.video_processor is not None:
        vp = ctx.video_processor
        vp.set_selected(selected_ex)
        vp.set_speak(speak_enabled, gate=speak_gate)
        vp.set_debug(show_debug)
        vp.speak_fn = tts.say

        tips_count_placeholder.metric("Tips this set", vp.tips_this_set)
        if end_clicked:
            if time.time() - ss.last_summary_ts < 5:
                (st.toast if hasattr(st, "toast") else st.info)("Summary just delivered.")
            else:
                summary = vp.speak_summary(k=3, min_count=2)
                if summary:
                    ss.last_summary_text = summary
                    ss.last_summary_ts = time.time()
                    (st.toast if hasattr(st, "toast") else st.info)("Speaking set summary…")
                    vp.reset_set()
                    tips_count_placeholder.metric("Tips this set", vp.tips_this_set)
                    ss.camera_paused = True
                    try: ctx.stop()
                    except Exception: pass
                    rerun()
                else:
                    (st.toast if hasattr(st, "toast") else st.info)("No tips collected this set.")
        payload = vp.consume_summary_payload()
        if payload:
            ss.last_summary_text = payload["text"]
            ss.last_summary_ts = payload["ts"]
            vp.reset_set()
            tips_count_placeholder.metric("Tips this set", vp.tips_this_set)
            if ss.auto_pause_on_summary:
                ss.camera_paused = True
                try: ctx.stop()
                except Exception: pass
                rerun()
        file_data = _read_summary_file()
        if file_data and float(file_data.get("ts", 0.0)) > float(ss.last_summary_ts or 0.0):
            ss.last_summary_text = str(file_data.get("text", "") or "")
            ss.last_summary_ts = float(file_data.get("ts", time.time()))
            vp.reset_set()
            tips_count_placeholder.metric("Tips this set", vp.tips_this_set)
            if ss.auto_pause_on_summary:
                ss.camera_paused = True
                try: ctx.stop()
                except Exception: pass
                rerun()
else:
    st.info("Camera paused. Press START to begin the next set.")
    if st.button("START camera", type="primary", use_container_width=True):
        ss.camera_paused = False
        rerun()

tts.render()
