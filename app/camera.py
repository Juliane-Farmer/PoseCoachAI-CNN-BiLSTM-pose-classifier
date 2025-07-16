import subprocess
import cv2
import mediapipe as mp
import numpy as np
from typing import Sequence, Dict
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
        subprocess.Popen(['espeak', '--stdin'], stdin=subprocess.PIPE)\
                 .communicate(input=text.encode())

def calculate_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """
    Compute the angle at point `b` formed by segments a–b and c–b.
    Returns angle in degrees.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

JOINTS = {
    "right_knee":  (mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
                    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    "left_knee":   (mp.solutions.pose.PoseLandmark.LEFT_HIP,
                    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
                    mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    "right_elbow": (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                    mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    "left_elbow":  (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                    mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    "right_shoulder": (mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                       mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                       mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    "left_shoulder":  (mp.solutions.pose.PoseLandmark.LEFT_HIP,
                       mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                       mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    "right_hip":   (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                    mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    "left_hip":    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.LEFT_HIP,
                    mp.solutions.pose.PoseLandmark.LEFT_KNEE),
}

def main() -> None:
    """Main loop: capture video, compute & display joint angles, speak cues."""
    mp_pose   = mp.solutions.pose
    mp_draw   = mp.solutions.drawing_utils
    engine    = None

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        last_states: Dict[str, str] = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                angles = {}
                for name, (A, B, C) in JOINTS.items():
                    a = [lm[A].x * w, lm[A].y * h]
                    b = [lm[B].x * w, lm[B].y * h]
                    c = [lm[C].x * w, lm[C].y * h]
                    angle = calculate_angle(a, b, c)
                    angles[name] = angle

                    cv2.putText(
                        img_bgr,
                        f"{name.split('_')[1]}:{int(angle)}°",
                        tuple(np.int32(b)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA
                    )

                avg_knee = (angles["right_knee"] + angles["left_knee"]) / 2
                state = ("down" if avg_knee < 70 else
                         "up"   if avg_knee > 160 else last_states.get("squat", "up"))
                if state != last_states.get("squat"):
                    speak(state.capitalize())
                    last_states["squat"] = state

            cv2.imshow("PoseCoachAI", img_bgr)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
