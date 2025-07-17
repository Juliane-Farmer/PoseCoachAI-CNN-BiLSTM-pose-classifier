
import argparse
import threading
import pyttsx3  
import time  

import cv2
import mediapipe as mp
import numpy as np
import logging
import subprocess
import platform


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

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
    "right_hip":    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                      mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                      mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    "left_hip":     (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                      mp.solutions.pose.PoseLandmark.LEFT_HIP,
                      mp.solutions.pose.PoseLandmark.LEFT_KNEE),
}

EXERCISES = {
    "squat":        {"joints": ["right_knee", "left_knee"],   "down_thresh": 70,  "up_thresh": 160},
    "lunge":        {"joints": ["right_knee", "left_knee"],   "down_thresh": 80,  "up_thresh": 170},
    "pushup":       {"joints": ["right_elbow", "left_elbow"], "down_thresh": 90,  "up_thresh": 160},
    "curl":         {"joints": ["right_elbow", "left_elbow"], "down_thresh": 45,  "up_thresh": 160},
    "situp":        {"joints": ["right_hip", "left_hip"],     "down_thresh": 100, "up_thresh": 160},
    "russian_twist":{"joints": ["right_shoulder", "left_shoulder"], "down_thresh": 50, "up_thresh": 130},}

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return float(angle)

def speak(text: str) -> None:
    if platform.system() == 'Windows':
        cmd = [
            'powershell', '-Command',
            f"Add-Type -AssemblyName System.Speech;"
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\"{text}\");"
        ]
        subprocess.Popen(cmd)
    else:
        subprocess.Popen(['espeak', '--stdout', text.encode()], stdout=subprocess.DEVNULL)


def main(selected_exercise: str) -> None:
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    last_state = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            angles = {}
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark

                for name, (A, B, C) in JOINTS.items():
                    a = np.array([lm[A].x * w, lm[A].y * h])
                    b = np.array([lm[B].x * w, lm[B].y * h])
                    c = np.array([lm[C].x * w, lm[C].y * h])
                    angle = calculate_angle(a, b, c)
                    angles[name] = angle
                    cv2.putText(
                        img_bgr,
                        f"{int(angle)}°",
                        tuple(b.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA, )
                params = EXERCISES[selected_exercise]
                joints = params['joints']
                avg_ang = sum(angles[j] for j in joints) / len(joints)
                down_th, up_th = params['down_thresh'], params['up_thresh']

                transition = None
                if last_state is None:
                    if avg_ang >= up_th:
                        last_state = 'up'
                    elif avg_ang <= down_th:
                        last_state = 'down'
                else:
                    logging.info(f"{selected_exercise} avg_angle={avg_ang:.1f}, state={last_state}")
                    if avg_ang <= down_th and last_state == 'up':
                        transition = 'Down'
                        last_state = 'down'
                    elif avg_ang >= up_th and last_state == 'down':
                        transition = 'Up'
                        last_state = 'up'

                if transition:
                    logging.info(f"Speaking: {transition}")
                    speak(transition)

            cv2.imshow('PoseCoachAI', img_bgr)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exercise',
        choices=list(EXERCISES.keys()),
        default='squat',
        help='Exercise to coach'
    )
    args = parser.parse_args()
    logging.info(f"Coaching exercise: {args.exercise}")
    main(args.exercise)
