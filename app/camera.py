
"""
PoseCoachAI camera module:
  - grabs webcam frames  
  - runs MediaPipe Pose  
  - overlays landmarks & knee-angle text  
  - speaks “Up”/“Down” cues via eSpeak or pyttsx3 on Windows
"""

import subprocess
import cv2
import mediapipe as mp
import numpy as np
from typing import Sequence
import platform
import pyttsx3


def speak(text: str) -> None:
    """Cross-platform TTS: pyttsx3 on Windows, espeak on Linux."""
    if platform.system() == "Windows":
        if not hasattr(speak, "_engine"):
            speak._engine = pyttsx3.init()
            speak._engine.setProperty("rate", 150)
        speak._engine.say(text)
        speak._engine.runAndWait()
    else:
        subprocess.Popen(
            ['espeak', '--stdin'],
            stdin=subprocess.PIPE
        ).communicate(input=text.encode())


def calculate_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """
    Compute the angle at point `b` formed by segments a–b and c–b.
    Returns:
        Angle in degrees.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main() -> None:
    """Main loop: capture video, detect pose, speak cues, and display overlay."""
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        last_state = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    img_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS )

                lm = results.pose_landmarks.landmark
                h, w = frame.shape[:2]
                hip = [
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h]
                knee = [
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h]
                ankle = [
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h]

                angle = calculate_angle(hip, knee, ankle)
                cv2.putText(
                    img_bgr,
                    f"{int(angle)}°",
                    tuple(np.int32(knee)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if angle < 70 and last_state != "down":
                    speak("Down")
                    last_state = "down"
                elif angle > 160 and last_state != "up":
                    speak("Up")
                    last_state = "up"

            cv2.imshow("PoseCoachAI", img_bgr)
            if cv2.waitKey(5) & 0xFF == 27: 
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
