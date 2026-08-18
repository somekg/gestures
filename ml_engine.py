from sklearn.svm import SVC
import pickle
import os

MODEL_PATH = 'models/my_custom_gestures.pkl'

def normalize_landmarks(hand_landmarks):
    """Converts 21 MediaPipe landmarks into a flat list of 63 relative coordinates."""
    # The wrist is always landmark 0
    base_x = hand_landmarks[0].x 
    base_y = hand_landmarks[0].y 
    base_z = hand_landmarks[0].z 
    
    normalized_list = [] 
    
    for landmark in hand_landmarks: 
        # Subtract the wrist position from every joint
        normalized_list.extend([ 
            landmark.x - base_x, 
            landmark.y - base_y, 
            landmark.z - base_z 
        ]) 
        
    return normalized_list 

def train_custom_gestures(X_data, y_labels):
    print("Training custom model...") 
    
    # --- SILENT BACKGROUND INJECTION ---
    # Create copies so we don't accidentally show 'Unknown' in the user's UI
    training_X = list(X_data)
    training_y = list(y_labels)
    
    background_path = 'models/base_background.pkl'
    if os.path.exists(background_path):
        with open(background_path, 'rb') as f:
            bg_X, bg_y = pickle.load(f)
            training_X.extend(bg_X)
            training_y.extend(bg_y)
            print(f"Silently injected {len(bg_X)} 'Unknown' background frames.")
    # -----------------------------------
    
    # Safety check: Do we have at least 2 classes after injection?
    unique_classes = set(training_y)
    if len(unique_classes) < 2:
        print("Cannot train: Need at least 2 distinct classes (including the background).")
        return None
        
    # Initialize the Support Vector Machine with probability enabled
    model = SVC(kernel='rbf', probability=True) 
    
    # Train the model 
    model.fit(training_X, training_y) 
    
    # Save the trained model to a file
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f: 
        pickle.dump(model, f) 
        
    print("Model trained and saved successfully!") 
    return model

def load_model():
    """Loads the custom Scikit-Learn model if available."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None