import cv2
import time
import os
import numpy as np
import mediapipe as mp

# --- NEW: NATIVE LINUX KEYBOARD INJECTION ---
try:
    from evdev import UInput, ecodes as e
    # Create the virtual keyboard
    virtual_keyboard = UInput()
except PermissionError:
    print("FATAL: Permission denied to create a virtual keyboard.")
    print("For testing, run with: sudo -E python main.py")
    exit()

# Map simple text to Linux kernel key codes
EVDEV_MAPPING = {
    "space": e.KEY_SPACE,
    "win": e.KEY_LEFTMETA,
    "volumemute": e.KEY_MUTE,
    "enter": e.KEY_ENTER,
    "a": e.KEY_A,
    "b": e.KEY_B,
    "c": e.KEY_C
}
# --------------------------------------------

from vision import (
    VisionTracker, draw_landmarks_on_image, 
    GestureRecognizer, GestureRecognizerOptions, 
    BaseOptions, VisionRunningMode
)
from ml_engine import normalize_landmarks, train_custom_gestures, load_model
from dataset_manager import load_dataset, save_dataset, remove_gesture

def main():
    os.makedirs('thumbnails', exist_ok=True)
    X_data, y_labels, macro_map = load_dataset()
    custom_classifier = load_model()
    current_gesture = "Neutral" 
    
    current_held_gesture = None
    held_start_time = 0
    HOLD_TIME_REQUIRED = 0.1 # Set to 1.0 to prevent ZeroDivisionError
    
    # --- MACRO TOGGLE ---
    macros_enabled = False # Start OFF so you can safely record/train
    
    vision_tracker = VisionTracker()
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='/home/alex/Desktop/gestures/gesture_recognizer.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=vision_tracker.result_callback,
        num_hands=1
    )

    cap = cv2.VideoCapture(0)

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            flipped_frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)

            latest_result = vision_tracker.get_latest_result()

            if custom_classifier is not None and latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    flat_data = normalize_landmarks(hand_landmarks)
                    prediction = custom_classifier.predict([flat_data])[0]
                    confidence = np.max(custom_classifier.predict_proba([flat_data]))
                    
                    if confidence > 0.80 and prediction != "Unknown":
                        cv2.putText(flipped_frame, f"Prediction: {prediction} ({confidence:.2f})", 
                                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # --- NATIVE CONTINUOUS MACRO EXECUTION ---
                        # ONLY FIRE IF MACROS ARE ENABLED!
                        if macros_enabled and prediction in macro_map:
                            current_time = time.time()
                            
                            if prediction == current_held_gesture:
                                elapsed_time = current_time - held_start_time
                                
                                if elapsed_time < HOLD_TIME_REQUIRED:
                                    # Still charging up to prevent accidental transition triggers
                                    hold_progress = elapsed_time / HOLD_TIME_REQUIRED
                                    cv2.putText(flipped_frame, f"Activating: {hold_progress*100:.0f}%", 
                                                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                                else:
                                    # Fully charged! Fire the key!
                                    cv2.putText(flipped_frame, f"FIRING: '{macro_map[prediction]}'", 
                                                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                                
                                    key_string = macro_map[prediction]
                                    if key_string in EVDEV_MAPPING:
                                        kernel_code = EVDEV_MAPPING[key_string]
                                        
                                        # Press down and release immediately
                                        virtual_keyboard.write(e.EV_KEY, kernel_code, 1) 
                                        virtual_keyboard.write(e.EV_KEY, kernel_code, 0) 
                                        virtual_keyboard.syn() 
                                        
                                        print(f"MACRO FIRED: Pressed '{key_string}'")
                                    
                                    # --- THE COOLDOWN FIX ---
                                    # Reset the start time so the cooldown equals the hold time!
                                    held_start_time = current_time
                                        
                            else:
                                # The gesture just changed, reset the timer
                                current_held_gesture = prediction
                                held_start_time = current_time
                        # -----------------------------------------
                    else:
                        current_held_gesture = None
            else:
                current_held_gesture = None
            
            if latest_result is not None:
                flipped_frame = draw_landmarks_on_image(flipped_frame, latest_result)

            # UI: Show Macro State
            macro_status_text = "Macros: ON (Live)" if macros_enabled else "Macros: PAUSED"
            macro_color = (0, 0, 255) if macros_enabled else (150, 150, 150)
            cv2.putText(flipped_frame, f"{macro_status_text} - Press 'e' to toggle", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, macro_color, 2)

            cv2.putText(flipped_frame, f"Target Label for SPACEBAR: '{current_gesture}'", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            key = cv2.waitKey(1) & 0xFF
            
            # ==========================================
            # 1. GLOBAL CONTROLS (Always Active)
            # ==========================================
            if key == ord('q'):
                save_dataset(X_data, y_labels, macro_map)
                break
                
            elif key == ord('e'):
                macros_enabled = not macros_enabled
                if macros_enabled:
                    print("\n--- MODE: MACROS LIVE ---")
                    print("Setup controls disabled. Gestures will now press keys!\n")
                else:
                    print("\n--- MODE: SETUP ---")
                    print("Macros paused. Safe to record, edit, and train.\n")
                    current_held_gesture = None # Reset trackers safely
            
            # ==========================================
            # 2. SETUP CONTROLS (Only Active if Macros OFF)
            # ==========================================
            elif not macros_enabled:
                
                if key == ord('n'):
                    print("\n---")
                    current_gesture = input("Terminal Input -> Enter name for new gesture: ").strip()
                    print(f"Assign a valid key from EVDEV_MAPPING (e.g., 'space', 'volumemute', 'win').")
                    macro_key = input("Terminal Input -> Enter key (leave blank for none): ").strip()
                    if macro_key:
                        macro_map[current_gesture] = macro_key
                        print(f"Assigned '{macro_key}' to '{current_gesture}'.")
                    print(f"Click back to the video window and hold SPACE to record.")
                    print("---\n")
                    
                elif key == ord(' '): 
                    if latest_result and latest_result.hand_landmarks:
                        if current_gesture not in y_labels:
                            blank_canvas = np.zeros(flipped_frame.shape, dtype=np.uint8)
                            thumbnail = draw_landmarks_on_image(blank_canvas, latest_result)
                            cv2.imwrite(f"thumbnails/{current_gesture}.jpg", thumbnail)

                        for hand_landmarks in latest_result.hand_landmarks:
                            flat_data = normalize_landmarks(hand_landmarks)
                            X_data.append(flat_data)
                            y_labels.append(current_gesture)
                            
                        cv2.putText(flipped_frame, f"Recording {current_gesture}: {y_labels.count(current_gesture)} frames", 
                                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                elif key == ord('m'):
                    print("\n--- UPDATE MACRO ---")
                    target = input("Terminal Input -> Enter the exact name of the existing gesture: ").strip()
                    if target in y_labels:
                        print(f"Assign a valid key from EVDEV_MAPPING.")
                        macro_key = input(f"Terminal Input -> Enter new key for '{target}' (leave blank to remove): ").strip()
                        if macro_key:
                            macro_map[target] = macro_key
                            print(f"Success! Assigned '{macro_key}' to '{target}'.")
                        else:
                            if target in macro_map:
                                del macro_map[target]
                                print(f"Removed macro assignment for '{target}'.")
                        save_dataset(X_data, y_labels, macro_map)
                    else:
                        print(f"Error: Gesture '{target}' does not exist in your dataset.")
                    print("---\n")
                    
                elif key == ord('r'):
                    print("\n---")
                    target = input("Terminal Input -> Enter the exact name of the gesture to remove: ").strip()
                    if target.lower() == "neutral":
                        print("WARNING: 'Neutral' cannot be permanently removed.")
                        confirm = input("Do you want to CLEAR it so you can retrain it from scratch? (yes/no): ").strip().lower()
                        if confirm == 'yes':
                            X_data, y_labels = remove_gesture(X_data, y_labels, "Neutral")
                            if "Neutral" in macro_map: del macro_map["Neutral"]
                    else:
                        X_data, y_labels = remove_gesture(X_data, y_labels, target)
                        if target in macro_map: del macro_map[target]
                    print("---\n")
                    
                elif key == ord('t'):
                    if "Neutral" not in y_labels:
                        print("Cannot train! You are required to have a 'Neutral' gesture.")
                    else:
                        save_dataset(X_data, y_labels, macro_map)
                        if len(X_data) > 0:
                            custom_classifier = train_custom_gestures(X_data, y_labels)
                        else:
                            custom_classifier = None

            cv2.imshow('MediaPipe Gesture Recognition', flipped_frame)

    cap.release()
    cv2.destroyAllWindows()
    # Close the virtual keyboard cleanly
    virtual_keyboard.close() 

if __name__ == "__main__":
    main()