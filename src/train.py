# src/train.py
import csv
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(csv_path, expected_features=136):
    """Load two-handed data: 136 features (68 per hand)"""
    X, y, labels = [], [], []
    skipped = 0
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None, None, None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                skipped += 1
                continue
            
            label = row[0]
            try:
                features = [float(v) for v in row[1:]]
                
                if len(features) != expected_features:
                    print(f"Row {i}: Expected {expected_features}, got {len(features)} - skipping")
                    skipped += 1
                    continue
                
                labels.append(label)
                X.append(features)
            except ValueError as e:
                print(f"Row {i}: Invalid data - {e}")
                skipped += 1
                continue
    
    if skipped > 0:
        print(f"⚠ Skipped {skipped} malformed rows")
    
    if not X:
        print("Error: No valid samples")
        return None, None, None
    
    print(f"✓ Loaded {len(X)} samples with {expected_features} features each")
    return X, labels, list(set(labels))

def main():
    csv_path = "data/samples.csv"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load two-handed data (136 features)
    X, labels, unique_labels = load_data(csv_path, expected_features=136)
    if X is None:
        return
    
    unique_labels = sorted(unique_labels)
    lab2idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx2lab = {i: lab for lab, i in lab2idx.items()}
    
    y = [lab2idx[lab] for lab in labels]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    print(f"\n✓ Dataset: {X.shape}")
    print(f"✓ Classes: {len(unique_labels)}")
    print(f"✓ Labels: {unique_labels}")
    
    if len(X) < 10:
        print("Error: Need at least 10 samples")
        return
    
    test_size = min(0.2, max(0.1, 1.0 / len(X)))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y if len(unique_labels) > 1 else None, random_state=42
    )
    
    print(f"\n✓ Train: {len(X_tr)}")
    print(f"✓ Test: {len(X_te)}")
    
    # Larger network for two-handed complexity
    print("\nTraining two-handed MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        random_state=42,
        verbose=True
    )
    
    clf.fit(X_tr, y_tr)
    
    train_acc = accuracy_score(y_tr, clf.predict(X_tr))
    test_acc = accuracy_score(y_te, clf.predict(X_te))
    
    print(f"\n✓ Train accuracy: {train_acc:.3f}")
    print(f"✓ Test accuracy: {test_acc:.3f}")
    
    if len(X_te) > 0:
        y_pred = clf.predict(X_te)
        print("\nClassification Report:")
        print(classification_report(y_te, y_pred, target_names=unique_labels, zero_division=0))
    
    model_path = os.path.join(model_dir, "isl_static_clf.joblib")
    labels_path = os.path.join(model_dir, "labels.json")
    
    joblib.dump(clf, model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(idx2lab, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Model: {model_path}")
    print(f"✓ Labels: {labels_path}")
    print("\n✓ Two-handed model ready!")

if __name__ == "__main__":
    main()
