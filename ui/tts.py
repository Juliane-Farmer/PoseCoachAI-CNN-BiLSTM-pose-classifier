import threading, queue, time, logging, sys

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import pythoncom
except Exception:
    pythoncom = None

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


class TTS:
  
    def __init__(self, prefer_browser: bool = False, rate: int = 180, volume: float = 1.0,
                 voice: str | None = None, beep: bool = True):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed; server TTS unavailable")

        self._q: queue.Queue[str | None] = queue.Queue()
        self._alive = True
        self._cfg = {"rate": rate, "volume": float(volume), "voice": voice}
        self._beep = bool(beep)
        self._prefer_browser = prefer_browser  

        self._worker = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._worker.start()
        LOG.info("ServerTTS worker started")

    def _run(self):
        engine = None
        if pythoncom is not None:
            try:
                pythoncom.CoInitialize()
            except Exception:
                pass

        try:
            try:
                engine = pyttsx3.init(driverName="sapi5")  
            except Exception:
                engine = pyttsx3.init()

            try:
                engine.setProperty("rate", self._cfg["rate"])
                engine.setProperty("volume", self._cfg["volume"])
                if self._cfg["voice"] is not None:
                    engine.setProperty("voice", self._cfg["voice"])
            except Exception:
                pass

            while self._alive:
                try:
                    text = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if text is None:
                    break

                try:
                    if self._beep and winsound is not None:
                        try:
                            winsound.Beep(1000, 120)  
                        except Exception:
                            pass

                    LOG.info(f"tts_start  | {text}")
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
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
            LOG.info("ServerTTS worker stopped")

    def say(self, text: str):
        """Flush queued-but-not-started items and enqueue `text` next."""
        if not text:
            return
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
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
