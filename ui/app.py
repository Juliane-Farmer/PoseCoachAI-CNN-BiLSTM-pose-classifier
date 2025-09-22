import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("absl_log_level", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit.components.v1 as components

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTO = True
except Exception:
    _HAS_AUTO = False

from posecoach.inference.engine import load_engine
from overlay_processor import OverlayProcessor, DEFAULT_SPEAK_GATE
from posecoach.tts import TTS

st.set_page_config(page_title="PoseCoachAI — Real-time Coach", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "outputs" / "session_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_root = logging.getLogger()
if not _root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],)

SUMMARY_PATH = ROOT / "outputs" / "session_logs" / "last_summary.json"
PAUSE_FLAG_PATH = ROOT / "outputs" / "session_logs" / "pause.flag"

def _read_summary_file():
    try:
        if SUMMARY_PATH.exists():
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _summary_mtime_ns() -> int:
    try:
        return SUMMARY_PATH.stat().st_mtime_ns
    except Exception:
        return 0

def _pause_flag_mtime_ns() -> int:
    try:
        return PAUSE_FLAG_PATH.stat().st_mtime_ns
    except Exception:
        return 0

ss = st.session_state
ss.setdefault("camera_paused", False)
ss.setdefault("webrtc_nonce", 0)
ss.setdefault("last_summary_text", "")
ss.setdefault("last_summary_ts", 0.0)
ss.setdefault("last_summary_mtime_ns", 0)
ss.setdefault("last_pause_flag_ns", _pause_flag_mtime_ns())
ss.setdefault("aggressive_stop", False)

if "init_done" not in ss:
    ss.init_done = True
    prev = _read_summary_file()
    if prev:
        try:
            ss.last_summary_ts = float(prev.get("ts", 0.0))
            ss.last_summary_text = str(prev.get("text", "") or "")
            ss.last_summary_mtime_ns = _summary_mtime_ns()
        except Exception:
            pass

def rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

@st.cache_resource(show_spinner=False)
def get_engine():
    return load_engine(ROOT)

@st.cache_resource(show_spinner=False)
def get_tts(cache_buster: int = 1):
    return TTS(prefer_browser=False, rate=170, volume=1.0, beep=False)

ENGINE = get_engine()
tts = get_tts()

if not ss.camera_paused:
    if _HAS_AUTO:
        st_autorefresh(interval=500, key=f"posecoach_tick_{ss.webrtc_nonce}")
    else:
        components.html(
            """
            <script>
              setTimeout(function(){
                try { window.parent.postMessage({type: 'streamlit:rerun'}, '*'); } catch(e) {}
              }, 500);
            </script>
            """,
            height=0,)

st.title("PoseCoachAI — Real-time Coach")
st.info("PoseCoachAI works best when you wear tight or fitted clothing. Avoid loose garments that hide joints.")

c1, c2 = st.columns([2, 1])
with c1:
    selected_ex = st.selectbox("Select exercise", options=ENGINE["type_names"], index=0)
with c2:
    speak_enabled = st.toggle("Voice tips", value=True)

with st.sidebar:
    st.subheader("Debug & Utilities")
    if st.button("Speak test"):
        tts.say("This is a PoseCoach test.")
    beep_debug = st.toggle("Beep (debug)", value=False)
    try:
        tts.set_beep(beep_debug)
    except Exception:
        pass
    if st.button("Beep test only"):
        try:
            tts.beep_once()
        except Exception:
            pass
    show_debug = st.toggle("Debug overlays", value=False)
    ss.aggressive_stop = st.toggle("Stop camera after summary (aggressive)", value=False)
    speak_gate = st.slider("Speak ≥", 0.0, 1.0, DEFAULT_SPEAK_GATE, 0.05)

st.markdown("### Session Controls")
t1 = st.columns([2])[0]
with t1:
    end_clicked = st.button("End set ▶ Speak summary", type="primary", use_container_width=True)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
ctx = None

def _pause_with_payload(vp, ctx_obj, payload=None, file_data=None):
    if payload:
        ss.last_summary_text = payload.get("text", "")
        ss.last_summary_ts = float(payload.get("ts", time.time()))
        ss.last_summary_mtime_ns = _summary_mtime_ns()
    elif file_data:
        ss.last_summary_text = str(file_data.get("text", "") or "")
        ss.last_summary_ts = float(file_data.get("ts", time.time()))
        ss.last_summary_mtime_ns = _summary_mtime_ns()
    try:
        tts.flush()
    except Exception:
        pass
    if vp is not None:
        try:
            vp.suppress_ready(True)
        except Exception:
            pass
        try:
            vp._request_pause = False
            vp.reset_set()
        except Exception:
            pass
    ss.camera_paused = True
    ss.webrtc_nonce += 1
    if ss.aggressive_stop:
        components.html(
            """
            <script>
            (function(){
              try{
                const vids = Array.from(document.querySelectorAll('video'));
                vids.forEach(v=>{
                  const s = v.srcObject;
                  if(s){ (s.getTracks()||[]).forEach(t=>{ try{t.stop()}catch(e){} }); }
                });
              }catch(e){}
            })();
            </script>
            """,
            height=0)
    try:
        if ctx_obj is not None:
            ctx_obj.stop()
    except Exception:
        ss.camera_paused = True
    rerun()

def render_summary_section():
    st.markdown("### Summary")
    b1, b2 = st.columns(2)
    with b1:
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
    with b2:
        if st.button("Clear last summary", use_container_width=True):
            ss.last_summary_text = ""
            ss.last_summary_ts = 0.0
            try:
                SUMMARY_PATH.unlink(missing_ok=True)
            except Exception:
                pass

if not ss.camera_paused:
    ctx = webrtc_streamer(
        key=f"posecoach_{ss.webrtc_nonce}",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": {"width": 640, "height": 360, "frameRate": 24}, "audio": False},
        video_processor_factory=lambda: OverlayProcessor(
            engine=ENGINE,
            selected=selected_ex,
            speak=speak_enabled,
            speak_gate=speak_gate,
            show_debug=show_debug,
            speak_fn=tts.say,
            summary_path=SUMMARY_PATH,
            pause_flag_path=PAUSE_FLAG_PATH, ),
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True)

    if ctx.video_processor is not None:
        vp = ctx.video_processor
        vp.set_selected(selected_ex)
        vp.set_speak(speak_enabled, gate=speak_gate)
        vp.set_debug(show_debug)
        vp.speak_fn = tts.say
        if getattr(vp, "_request_pause", False):
            payload = vp.consume_summary_payload()
            file_data = _read_summary_file()
            _pause_with_payload(vp, ctx, payload=payload, file_data=file_data)
        if end_clicked:
            try:
                vp.suppress_ready(True)
            except Exception:
                pass
            summary = vp.speak_summary(k=3, min_count=2)
            if summary:
                ss.last_summary_text = summary
                ss.last_summary_ts = time.time()
                ss.last_summary_mtime_ns = _summary_mtime_ns()
                _pause_with_payload(vp, ctx)
            else:
                st.info("No tips collected this set.")
        payload = vp.consume_summary_payload()
        if payload:
            _pause_with_payload(vp, ctx, payload=payload)
        file_data = _read_summary_file()
        file_mtime_ns = _summary_mtime_ns()
        if file_data and file_mtime_ns > (ss.last_summary_mtime_ns or 0):
            _pause_with_payload(vp, ctx, file_data=file_data)
        pf_mtime = _pause_flag_mtime_ns()
        if pf_mtime > (ss.last_pause_flag_ns or 0):
            ss.last_pause_flag_ns = pf_mtime
            _pause_with_payload(vp, ctx)
    render_summary_section()
else:
    st.info("Camera paused. Press START to begin the next set.")
    if st.button("START camera", type="primary", use_container_width=True):
        ss.camera_paused = False
        ss.webrtc_nonce += 1
        rerun()
    render_summary_section()

tts.render()
