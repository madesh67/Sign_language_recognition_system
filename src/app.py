# src/app.py
import cv2
import json
import joblib
import numpy as np
import mediapipe as mp

LABELS = {int(k): v for k, v in json.load(open("models/labels.json")).items()}
clf = joblib.load("models/isl_static_clf.joblib")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_hand_features(hand_landmarks, handedness, frame_shape):
    h, w = frame_shape[:2]
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    landmark_vec = np.array(pts, dtype=np.float32).flatten()
    
    wrist = hand_landmarks.landmark[0]
    center_x, center_y = wrist.x, wrist.y
    
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    hand_label = 1.0 if handedness == "Right" else 0.0
    position_vec = np.array([center_x, center_y, width, height, hand_label], dtype=np.float32)
    
    return np.concatenate([landmark_vec, position_vec])

def hand_vec_two_hands(results, frame_shape):
    feature_size = 68
    
    if not results.multi_hand_landmarks:
        return np.zeros(feature_size * 2, dtype=np.float32)
    
    hands_data = []
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        hand_label = handedness.classification[0].label
        features = extract_hand_features(hand_landmarks, hand_label, frame_shape)
        hands_data.append((hand_label, features))
    
    hands_data.sort(key=lambda x: x[0])
    
    left_hand = np.zeros(feature_size, dtype=np.float32)
    right_hand = np.zeros(feature_size, dtype=np.float32)
    
    for hand_label, features in hands_data:
        if hand_label == "Left":
            left_hand = features
        else:
            right_hand = features
    
    return np.concatenate([left_hand, right_hand])

def predict(vec):
    proba = clf.predict_proba(vec)[0]
    idx = int(np.argmax(proba))
    return LABELS[idx], float(proba[idx])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                    min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        label, conf = "", 0.0
        hand_count = 0
        
        if res.multi_hand_landmarks:
            hand_count = len(res.multi_hand_landmarks)
            for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label
                
                wrist = hand_landmarks.landmark[0]
                h, w = frame.shape[:2]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, hand_label, (cx - 30, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            vec = hand_vec_two_hands(res, frame.shape).reshape(1, -1)
            label, conf = predict(vec)
        
        cv2.putText(frame, f"{label} ({conf:.2f}) | Hands: {hand_count}",
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.imshow("ISL Two-Handed Recognizer", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
