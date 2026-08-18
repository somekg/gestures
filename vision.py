import mediapipe as mp
import numpy as np

# Drawing Utilities Setup
mp_drawing = mp.tasks.vision.drawing_utils 
mp_drawing_styles = mp.tasks.vision.drawing_styles 
mp_hands = mp.tasks.vision.HandLandmarksConnections 

# Core Tasks API Setup
BaseOptions = mp.tasks.BaseOptions 
GestureRecognizer = mp.tasks.vision.GestureRecognizer 
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions 
VisionRunningMode = mp.tasks.vision.RunningMode 

class VisionTracker:
    def __init__(self):
        self.latest_result = None
        
    def result_callback(self, result, output_image, timestamp_ms):
        """Callback function triggered in the background."""
        self.latest_result = result
        
    def get_latest_result(self):
        return self.latest_result

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws the hand skeleton using the modern Tasks API."""
    if not detection_result or not detection_result.hand_landmarks: 
        return rgb_image 

    annotated_image = np.copy(rgb_image) 

    for hand_landmarks in detection_result.hand_landmarks: 
        mp_drawing.draw_landmarks( 
            annotated_image, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS, 
            mp_drawing_styles.get_default_hand_landmarks_style(), 
            mp_drawing_styles.get_default_hand_connections_style() 
        ) 
        
    return annotated_image 