# src/capture.py
import argparse
import os
import json
import csv
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_hand_features(hand_landmarks, handedness, frame_shape):
    """Extract features: landmarks + position + handedness"""
    h, w = frame_shape[:2]
    
    # 21 landmarks × (x,y,z) = 63 features per hand
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    landmark_vec = np.array(pts, dtype=np.float32).flatten()  # 63 values
    
    # Compute hand center (palm center - landmark 0 is wrist)
    wrist = hand_landmarks.landmark[0]
    center_x, center_y = wrist.x, wrist.y
    
    # Compute hand bounding box size (normalized)
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    # Handedness: 0=Left, 1=Right
    hand_label = 1.0 if handedness == "Right" else 0.0
    
    # Position features: [landmarks(63), center_x, center_y, width, height, handedness]
    position_vec = np.array([center_x, center_y, width, height, hand_label], dtype=np.float32)
    
    return np.concatenate([landmark_vec, position_vec])  # 68 features per hand

def hand_vec_two_hands(results, frame_shape):
    """Extract features for both hands or pad if only one hand detected"""
    feature_size = 68  # 63 landmarks + 5 position/handedness
    
    if not results.multi_hand_landmarks:
        # No hands: return zeros
        return np.zeros(feature_size * 2, dtype=np.float32)
    
    hands_data = []
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        hand_label = handedness.classification[0].label
        features = extract_hand_features(hand_landmarks, hand_label, frame_shape)
        hands_data.append((hand_label, features))
    
    # Sort: Left hand first, Right hand second for consistency
    hands_data.sort(key=lambda x: x[0])  # "Left" < "Right" alphabetically
    
    # Build final feature vector
    left_hand = np.zeros(feature_size, dtype=np.float32)
    right_hand = np.zeros(feature_size, dtype=np.float32)
    
    for hand_label, features in hands_data:
        if hand_label == "Left":
            left_hand = features
        else:
            right_hand = features
    
    # Concatenate: [left_hand(68), right_hand(68)] = 136 features
    return np.concatenate([left_hand, right_hand])

def save_reference_image(results, label, reference_dir="reference_signs", img_size=600):
    """Save ONLY the hand landmark pattern(s) on clean background"""
    label_dir = os.path.join(reference_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    existing = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]
    img_num = len(existing) + 1
    
    # Create blank white canvas
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    if not results.multi_hand_landmarks:
        return None
    
    # Collect all landmarks from all hands
    all_x, all_y = [], []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_x.append(lm.x)
            all_y.append(lm.y)
    
    # Calculate bounding box for all hands
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add padding
    padding = 0.15
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    
    def scale_point(x, y):
        px = int((x - x_min) / (x_max - x_min) * (img_size - 40) + 20)
        py = int((y - y_min) / (y_max - y_min) * (img_size - 40) + 20)
        return px, py
    
    # Draw each hand with different colors
    colors = [(0, 200, 0), (200, 0, 200)]  # Green for first, Magenta for second
    
    for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
        color = colors[i % len(colors)]
        hand_label = handedness.classification[0].label
        
        # Draw connections
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_pt = hand_landmarks.landmark[start_idx]
            end_pt = hand_landmarks.landmark[end_idx]
            
            start_x, start_y = scale_point(start_pt.x, start_pt.y)
            end_x, end_y = scale_point(end_pt.x, end_pt.y)
            
            cv2.line(canvas, (start_x, start_y), (end_x, end_y), color, 3)
        
        # Draw landmark points
        for lm in hand_landmarks.landmark:
            cx, cy = scale_point(lm.x, lm.y)
            cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(canvas, (cx, cy), 7, (0, 0, 0), 1)
        
        # Label the hand
        first_pt = hand_landmarks.landmark[0]
        label_x, label_y = scale_point(first_pt.x, first_pt.y)
        cv2.putText(canvas, hand_label[0], (label_x - 20, label_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    img_path = os.path.join(label_dir, f"{label}_{img_num:04d}.jpg")
    cv2.imwrite(img_path, canvas)
    return img_path

def parse_labels(labels_json: str, labels_csv: str):
    if labels_json and os.path.exists(labels_json):
        with open(labels_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = sorted(map(int, data.keys()))
            return [data[str(k)] for k in keys]
        return list(map(str, data))
    if labels_csv:
        return [s.strip() for s in labels_csv.split(",") if s.strip()]
    return ["HELLO","THANK_YOU","PLEASE","YES","NO","WATER","EAT","DRINK","A","B","C","ONE","TWO","THREE"]

def ensure_output_csv(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(path):
        open(path, "w", newline="", encoding="utf-8").close()

def main():
    ap = argparse.ArgumentParser(description="Two-handed ISL capture with position tracking")
    ap.add_argument("--labels-json", type=str, default="", help="Path to labels.json")
    ap.add_argument("--labels", type=str, default="", help="Comma-separated labels")
    ap.add_argument("--output", type=str, default="data/samples.csv", help="Output CSV")
    ap.add_argument("--samples-per-label", type=int, default=100, help="Target samples per label")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--mirror", action="store_true", help="Mirror preview")
    ap.add_argument("--min-det", type=float, default=0.6, help="min_detection_confidence")
    ap.add_argument("--min-trk", type=float, default=0.6, help="min_tracking_confidence")
    ap.add_argument("--save-references", action="store_true", help="Save reference images")
    ap.add_argument("--reference-dir", type=str, default="reference_signs", help="Reference directory")
    ap.add_argument("--ref-interval", type=int, default=10, help="Save reference every N captures")
    args = ap.parse_args()

    labels = parse_labels(args.labels_json, args.labels)
    if not labels:
        print("No labels provided. Use --labels or --labels-json")
        return
    
    print(f"✓ Two-handed mode enabled (136 features: 68 per hand)")
    print(f"Will capture {args.samples_per_label} samples for: {labels}")
    if args.save_references:
        print(f"Reference images → {args.reference_dir}/")
    
    counts = {lab: 0 for lab in labels}
    idx = 0
    ensure_output_csv(args.output)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera")
        return

    # Enable detection of up to 2 hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=args.min_det,
                        min_tracking_confidence=args.min_trk) as hands, \
         open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        while True:
            if all(counts[lab] >= args.samples_per_label for lab in labels):
                print("\n✓ All labels reached target!")
                break

            while counts[labels[idx]] >= args.samples_per_label:
                idx = (idx + 1) % len(labels)

            lab = labels[idx]

            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # Draw all detected hands
            hand_count = 0
            if res.multi_hand_landmarks:
                hand_count = len(res.multi_hand_landmarks)
                for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    hand_label = handedness.classification[0].label
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Label each hand
                    wrist = hand_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, hand_label, (cx - 30, cy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # HUD
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
            msg1 = f"Label: {lab}  [{counts[lab]}/{args.samples_per_label}]"
            msg2 = f"Hands detected: {hand_count}  (supports 0-2 hands)"
            msg3 = "c/SPACE=Capture  ]=Next  [=Prev  q=Quit"
            cv2.putText(frame, msg1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, msg2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            cv2.putText(frame, msg3, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("ISL Two-Handed Capture", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                print("\nQuitting...")
                break
            elif k in (ord('c'), ord(' ')):
                vec = hand_vec_two_hands(res, frame.shape).tolist()
                writer.writerow([lab] + vec)
                counts[lab] += 1
                print(f"✓ Captured {lab}: {counts[lab]}/{args.samples_per_label} ({hand_count} hands)")
                
                if args.save_references and counts[lab] % args.ref_interval == 0:
                    ref_path = save_reference_image(res, lab, args.reference_dir)
                    if ref_path:
                        print(f"  → Reference saved: {ref_path}")
                
                if counts[lab] >= args.samples_per_label:
                    print(f"✓ Completed {lab}!")
                    idx = (idx + 1) % len(labels)
            elif k == ord(']'):
                idx = (idx + 1) % len(labels)
                print(f"→ {labels[idx]}")
            elif k == ord('['):
                idx = (idx - 1) % len(labels)
                print(f"← {labels[idx]}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Data: {args.output}")
    print(f"Total: {sum(counts.values())} samples")

if __name__ == "__main__":
    main()
