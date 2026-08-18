import pickle
import os

DATASET_PATH = 'models/gesture_dataset.pkl'

def load_dataset():
    """Loads raw X_data, y_labels, and the macro map from disk if they exist."""
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'rb') as f:
            data = pickle.load(f)
            # Legacy support: If loading an older dataset that didn't have macros yet
            if len(data) == 2: 
                return data[0], data[1], {}
            # Modern support: Return all three
            return data[0], data[1], data[2] 
            
    return [], [], {}

def save_dataset(X_data, y_labels, macro_map):
    """Saves the raw dataset and macro map to disk to persist across sessions."""
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, 'wb') as f:
        pickle.dump((X_data, y_labels, macro_map), f)
    print(f"Dataset saved! Total frames: {len(X_data)}")

def remove_gesture(X_data, y_labels, gesture_name):
    """Filters out all instances of a specific gesture from the dataset."""
    filtered_X = []
    filtered_y = []
    
    for x, y in zip(X_data, y_labels):
        if y != gesture_name:
            filtered_X.append(x)
            filtered_y.append(y)
            
    return filtered_X, filtered_y