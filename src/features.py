# src/features.py
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def hand_vec(hand_landmarks):
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    arr = np.array(pts, dtype=np.float32).flatten()
    # Optional: wrist-relative normalization if used during training
    return arr
