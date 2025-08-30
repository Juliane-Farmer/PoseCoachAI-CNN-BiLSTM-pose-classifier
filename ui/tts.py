from __future__ import annotations
import uuid, pathlib
import streamlit as st
import streamlit.components.v1 as components


class BrowserTTS:
    def __init__(self, html_path: str):
        self.html_path = html_path
        ss = st.session_state
        ss.setdefault("tts_queue", [])
        ss.setdefault("tts_rate", 1.0)
        ss.setdefault("tts_pitch", 1.0)
        ss.setdefault("tts_volume", 1.0)
        ss.setdefault("tts_voice_pref", "")

    def enqueue(self, text: str):
        if text and text.strip():
            st.session_state["tts_queue"].append({"token": str(uuid.uuid4()), "text": text})

    def _fill(self, text: str, token: str) -> str:
        html = pathlib.Path(self.html_path).read_text(encoding="utf-8")
        repl = {
            "[[TEXT]]": (text or "").replace("\\", "\\\\").replace('"', '\\"'),
            "[[TOKEN]]": token,
            "[[RATE]]": str(st.session_state["tts_rate"]),
            "[[PITCH]]": str(st.session_state["tts_pitch"]),
            "[[VOLUME]]": str(st.session_state["tts_volume"]),
            "[[VOICE_SUBSTR]]": st.session_state["tts_voice_pref"].replace('"', '\\"'), }
        for k, v in repl.items():
            html = html.replace(k, v)
        return html

    def render(self):
        q = st.session_state.get("tts_queue", [])
        if q:
            item = q.pop(0)
            html = self._fill(item["text"], item["token"])
        else:
            html = self._fill("", "noop")
        components.html(html, height=0, scrolling=False, key="browser-tts")

class ServerTTS:
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 175)
        except Exception:
            self.engine = None

    def enqueue(self, text: str):
        if not text or not text.strip() or self.engine is None: return
        try: self.engine.say(text); self.engine.runAndWait()
        except Exception: pass

    def render(self): pass

class TTS:
    def __init__(self, prefer_browser: bool = True, html_path: str = "app/components/browser_tts.html"):
        self.browser = BrowserTTS(html_path) if prefer_browser else None
        self.server  = ServerTTS()

    def controls(self):
        with st.sidebar.expander("Voice (browser TTS)", expanded=False):
            st.session_state["tts_rate"] = st.slider("Rate", 0.5, 2.0, st.session_state["tts_rate"], 0.05)
            st.session_state["tts_pitch"] = st.slider("Pitch", 0.5, 1.5, st.session_state["tts_pitch"], 0.05)
            st.session_state["tts_volume"] = st.slider("Volume", 0.0, 1.0, st.session_state["tts_volume"], 0.05)
            st.session_state["tts_voice_pref"] = st.text_input("Voice contains… (name/lang)", st.session_state["tts_voice_pref"])

    def say(self, text: str):
        if self.browser is not None: self.browser.enqueue(text)
        else: self.server.enqueue(text)

def render(self):
    q = st.session_state.get("tts_queue", [])
    if q:
        item = q.pop(0)
        html = self._fill_template(item["text"], item["token"])
    else:
        html = self._fill_template("", "noop")
    components.html(html, height=1, scrolling=False)

