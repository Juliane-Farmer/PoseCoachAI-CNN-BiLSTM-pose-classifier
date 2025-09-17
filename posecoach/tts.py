import threading, queue, time, logging, sys

try:
    import pythoncom
except Exception:
    pythoncom = None

try:
    import win32com.client as win32client
except Exception:
    win32client = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import winsound
except Exception:
    winsound = None

LOG = logging.getLogger("posecoach.tts")
if not LOG.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)
LOG.propagate = False

class TTS:
    def __init__(self, prefer_browser: bool = False,rate: int = 175,volume: float = 1.0, voice: str | None = None,beep: bool = True, backend: str = "auto"):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._alive = True
        self._cfg = {"rate": int(rate), "volume": float(volume), "voice": voice}
        self._beep = bool(beep)
        self._prefer_browser = prefer_browser
        self._requested_backend = backend
        self._worker = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._worker.start()
        LOG.info("TTS worker starting…")

    def set_beep(self, enabled: bool) -> None:
        self._beep = bool(enabled)
        LOG.info(f"TTS beep={'on' if self._beep else 'off'}")

    def beep_once(self, freq: int = 1000, dur_ms: int = 80) -> None:
        if winsound is not None:
            try:
                winsound.Beep(freq, dur_ms)
            except Exception:
                pass

    def flush(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _map_wpm_to_sapi_rate(self, wpm: int) -> int:
        return max(-10, min(10, int(round((wpm - 175) / 25))))

    def _setup_sapi(self):
        if pythoncom is None or win32client is None:
            raise RuntimeError("SAPI COM not available")
        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
        voice = win32client.Dispatch("SAPI.SpVoice")
        try:
            tokens = voice.GetVoices()
            names = []
            chosen = None
            pref = (self._cfg["voice"] or "").lower().strip()
            for i in range(tokens.Count):
                desc = tokens.Item(i).GetDescription()
                names.append(desc)
                if pref and pref in desc.lower():
                    chosen = tokens.Item(i)
            LOG.info("SAPI voices: " + " | ".join(names))
            if chosen is not None:
                voice.Voice = chosen
                LOG.info(f"SAPI voice selected: {chosen.GetDescription()}")
        except Exception:
            pass
        try:
            voice.Rate = self._map_wpm_to_sapi_rate(self._cfg["rate"])
            voice.Volume = max(0, min(100, int(self._cfg["volume"] * 100)))
        except Exception:
            pass
        return voice

    def _setup_pyttsx3(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 not available")
        try:
            engine = pyttsx3.init(driverName="sapi5")
        except Exception:
            engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self._cfg["rate"])
            engine.setProperty("volume", self._cfg["volume"])
            if self._cfg["voice"]:
                for v in engine.getProperty("voices") or []:
                    name = (getattr(v, "name", "") or "") + " " + (getattr(v, "id", "") or "")
                    if self._cfg["voice"].lower() in name.lower():
                        engine.setProperty("voice", v.id)
                        LOG.info(f"pyttsx3 voice selected: {name.strip()}")
                        break
        except Exception:
            pass
        return engine

    def _run(self):
        backend = None
        sapi_voice = None
        engine = None
        try:
            choice = self._requested_backend.lower()
            if choice in ("auto", "sapi"):
                try:
                    sapi_voice = self._setup_sapi()
                    backend = "sapi"
                    LOG.info("TTS backend: SAPI (native COM)")
                except Exception as e:
                    LOG.warning(f"SAPI init failed ({e}); trying pyttsx3…")
            if backend is None:
                engine = self._setup_pyttsx3()
                backend = "pyttsx3"
                LOG.info("TTS backend: pyttsx3")
            while self._alive:
                try:
                    text = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if text is None:
                    break
                if self._beep and winsound is not None:
                    try:
                        winsound.Beep(1000, 80)
                    except Exception:
                        pass
                try:
                    LOG.info(f"tts_start  | {text}")
                    if backend == "sapi":
                        sapi_voice.Speak(text, 0)
                    else:
                        engine.say(text)
                        engine.runAndWait()
                    LOG.info(f"tts_played | {text}")
                    time.sleep(0.05)
                except Exception as e:
                    LOG.error(f"TTS error: {e}")
        finally:
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass
            if sapi_voice is not None and pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
            LOG.info("TTS worker stopped")

    def say(self, text: str):
        if not text:
            return
        self.flush()
        self._q.put(text)

    def controls(self):
        return None

    def render(self):
        return None

    def close(self):
        self._alive = False
        try:
            self._q.put(None)
        except Exception:
            pass
        try:
            self._worker.join(timeout=1.0)
        except Exception:
            pass

