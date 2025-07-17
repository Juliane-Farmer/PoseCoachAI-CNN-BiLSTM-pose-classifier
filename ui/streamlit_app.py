
import av
import cv2
import requests
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

API_URL = "http://127.0.0.1:8000/metrics"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class OverlayProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            metrics = requests.get(API_URL, timeout=0.5).json()
        except Exception:
            return frame

        y = 20
        for name, angle in metrics.items():
            if angle is None:
                continue
            cv2.putText(
                img,
                f"{name}: {int(angle)}°",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,)
            y += 20
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("PoseCoachAI Live Feedback")
st.write("Real-time joint-angle overlay (voice cues in camera.py)")

webrtc_streamer(
    key="pose",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=OverlayProcessor,
    rtc_configuration=RTC_CONFIGURATION,)
