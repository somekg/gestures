import math
from sklearn.svm import SVC, OneClassSVM
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os

MODEL_PATH = 'models/my_custom_gestures.pkl'

def normalize_landmarks(hand_landmarks):
    """Converts 21 MediaPipe landmarks into a flat, translation/scale invariant list."""
    # 1. Translation: The wrist is always landmark 0
    base_x = hand_landmarks[0].x 
    base_y = hand_landmarks[0].y 
    base_z = hand_landmarks[0].z 
    
    normalized_list = [] 
    
    for landmark in hand_landmarks: 
        # Translate to origin (NO ROTATION)
        tx = landmark.x - base_x
        ty = landmark.y - base_y
        tz = landmark.z - base_z
        
        normalized_list.extend([tx, ty, tz]) 
    
    # 2. Scale: Find the maximum absolute value in the flat list
    max_value = max(map(abs, normalized_list))
    
    # Divide every element by that max value (preventing division by zero)
    if max_value > 0.0:
        normalized_list = [n / max_value for n in normalized_list]
        
    return normalized_list

def train_custom_gestures(X_data, y_labels):
    print("Training custom model pipeline...")

    if len(set(y_labels)) < 2:
        print("Cannot train: Need at least 2 distinct classes.")
        return None

    # 1. Train the Gatekeeper (One-Class SVM on all valid user frames)
    # nu=0.05 sets a tight boundary with 5% margin for variance
    gatekeeper = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    gatekeeper.fit(X_data)

    # 2. Train the Multi-Class Classifier
    base_svc = SVC(kernel='rbf')
    classifier = CalibratedClassifierCV(base_svc, ensemble=False)
    classifier.fit(X_data, y_labels)

    # Bundle both into a single payload
    model_bundle = {
        'gatekeeper': gatekeeper,
        'classifier': classifier
    }

    # Save to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_bundle, f)

    print("Pipeline trained and saved successfully (Gatekeeper + Classifier)!")
    return model_bundle

def load_model():
    """Loads the model bundle from disk if available."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None