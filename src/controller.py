import os
import sys

class SuppressCWarnings:
    """Temporarily forces C/C++ level stderr output to /dev/null."""
    def __enter__(self):
        # Save a copy of the original OS-level stderr (file descriptor 2)
        self.original_stderr = os.dup(sys.stderr.fileno())
        # Open a pipeline to nowhere
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        # Force the OS to redirect all stderr to the void
        os.dup2(self.devnull, sys.stderr.fileno())

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the real stderr so normal Python errors show up again
        os.dup2(self.original_stderr, sys.stderr.fileno())
        os.close(self.devnull)
        os.close(self.original_stderr)

import cv2
import time
import numpy as np
import mediapipe as mp

from src.view import GestureUI, TerminalView
from src.input_handler import InputHandler
from src.vision import (
    VisionTracker, GestureRecognizer, 
    GestureRecognizerOptions, BaseOptions, VisionRunningMode
)
from src.ml_engine import normalize_landmarks, train_custom_gestures, load_model
from src.dataset_manager import load_dataset, save_dataset, remove_gesture

class AppController:
    def __init__(self):
        self.ui = GestureUI()
        self.text_view = TerminalView() # Instantiate the modular text view
        self.input_handler = InputHandler()
        self.X_data, self.y_labels, self.macro_map = load_dataset()
        self.custom_classifier = load_model()
        
        self.state = {
            'macros_enabled': False,
            'current_training_gesture': "None",
            'overlay_msg': "",
            'action_msg': "",
            'msg_color': (0, 0, 0),
            'allowed_hands': ["Left"] 
        }
        
        self.current_held_gesture = None
        self.held_start_time = 0
        self.last_fired_times = {} 
        self.is_running = True

        self.MIN_CONFIDENCE_MACRO = 0.95

    # ==========================================
    # 1. CORE LOOP & ORCHESTRATION
    # ==========================================
    def run(self):
        vision_tracker = VisionTracker()
        
        with SuppressCWarnings():
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=vision_tracker.result_callback,
                num_hands=2
            )

            cap = cv2.VideoCapture(0)

            with GestureRecognizer.create_from_options(options) as recognizer:
                while cap.isOpened() and self.is_running:
                    success, frame = cap.read()
                    if not success: break

                    flipped_frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    timestamp_ms = int(time.time() * 1000)
                    recognizer.recognize_async(mp_image, timestamp_ms)

                    latest_result = vision_tracker.get_latest_result()
                    
                    self.state['overlay_msg'] = ""
                    self.state['action_msg'] = ""

                    self._process_ai_predictions(latest_result)
                    key = self.ui.render(flipped_frame, self.state, latest_result)
                    self._route_keyboard_input(key, latest_result)

            cap.release()
            cv2.destroyAllWindows()

    # ==========================================
    # 2. AI & MACRO PROCESSING 
    # ==========================================
    def _process_ai_predictions(self, latest_result):
        # 1. BREAK CONDITION: Hand leaves the screen
        if self.custom_classifier is None or not latest_result or not latest_result.hand_landmarks:
            if self.current_held_gesture is not None:
                self.input_handler.release_all() # ---> SAFETY RELEASE
            self.current_held_gesture = None
            return

        for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
            hand_label = latest_result.handedness[idx][0].category_name
            if hand_label not in self.state['allowed_hands']:
                continue
                
            flat_data = normalize_landmarks(hand_landmarks)
            gatekeeper = self.custom_classifier['gatekeeper']
            classifier = self.custom_classifier['classifier']
            
            is_known_gesture = gatekeeper.predict([flat_data])[0]
            
            # 2. BREAK CONDITION: Gesture becomes "Unknown"
            if is_known_gesture != 1:
                self.state['overlay_msg'] = "Unknown Gesture"
                self.state['msg_color'] = (0, 0, 255)
                if self.current_held_gesture is not None:
                    self.input_handler.release_all() # ---> SAFETY RELEASE
                self.current_held_gesture = None
                continue

            prediction = classifier.predict([flat_data])[0]
            confidence = np.max(classifier.predict_proba([flat_data]))
       
            self.state['overlay_msg'] = f"Prediction: {prediction} ({confidence:.2f})"
            
            # 3. BREAK CONDITION: Confidence drops below threshold
            if confidence < self.MIN_CONFIDENCE_MACRO:
                self.state['msg_color'] = (0, 0, 255)  # Red
                if self.current_held_gesture is not None:
                    self.input_handler.release_all() # ---> SAFETY RELEASE
                self.current_held_gesture = None
                continue
            else:
                self.state['msg_color'] = (0, 255, 0)  # Green 
            
            if not self.state['macros_enabled'] or prediction not in self.macro_map:
                continue
                
            macro_data = self.macro_map[prediction]
            if isinstance(macro_data, str):
                macro_data = {'keys': macro_data, 'is_hold': False, 'hold_time': 0.1, 'cooldown': 1.0}
                
            current_time = time.time()
            
            if not macro_data['is_hold']:
                last_fired = self.last_fired_times.get(prediction, 0)
                if current_time - last_fired < macro_data['cooldown']:
                    remaining = macro_data['cooldown'] - (current_time - last_fired)
                    self.state['action_msg'] = f"COOLDOWN: {remaining:.1f}s"
                    continue

            # 4. BREAK CONDITION: Switching to a new valid gesture
            if prediction != self.current_held_gesture:
                self.input_handler.release_all() # ---> SAFETY RELEASE
                self.current_held_gesture = prediction
                self.held_start_time = current_time
                continue
                
            elapsed_time = current_time - self.held_start_time
            if elapsed_time < macro_data['hold_time']:
                hold_progress = elapsed_time / macro_data['hold_time'] if macro_data['hold_time'] > 0 else 1
                self.state['action_msg'] = f"Activating: {hold_progress*100:.0f}%"
                continue

            macro_keys = macro_data['keys']
            
            if macro_data['is_hold']:
                self.state['action_msg'] = f"HOLDING: '{macro_keys}'"
                self.input_handler.press_macro(macro_keys)
            else:
                self.state['action_msg'] = f"FIRING: '{macro_keys}'"
                self.input_handler.tap_macro(macro_keys)
            
            self.last_fired_times[prediction] = current_time
            if not macro_data['is_hold']:
                self.held_start_time = current_time

    # ==========================================
    # 3. EVENT ROUTER
    # ==========================================
    def _route_keyboard_input(self, key, latest_result):
        if key == ord('q'):
            self.cmd_quit()
        elif key == ord('e'):
            self.cmd_toggle_macros()
            
        if self.state['macros_enabled']:
            return

        if key == ord('n'):
            self.cmd_new_gesture()
        elif key == ord(' '):
            self.cmd_record_frame(latest_result)
        elif key == ord('m'):
            self.cmd_modify_macro()
        elif key == ord('r'):
            self.cmd_remove_gesture()
        elif key == ord('t'):
            self.cmd_train_model()
        elif key == ord('l'):
            self.cmd_list_gestures()
        elif key == ord('s'):
            self.cmd_training_status()

    # ==========================================
    # 4. COMMANDS 
    # ==========================================
    def cmd_quit(self):
        # save_dataset(self.X_data, self.y_labels, self.macro_map)
        self.is_running = False

    def cmd_toggle_macros(self):
        self.state['macros_enabled'] = not self.state['macros_enabled']
        if self.state['macros_enabled']:
            self.text_view.show_message("\n--- MODE: MACROS LIVE ---")
        else:
            self.text_view.show_message("\n--- MODE: SETUP ---")
            self.current_held_gesture = None
            self.state['current_training_gesture'] = "None"

    def cmd_list_gestures(self):
        unique_gestures = sorted(list(set(self.y_labels)))
        # We must format the dictionary before passing it to the text_view
        formatted_map = {}
        for k, v in self.macro_map.items():
            if isinstance(v, dict):
                m_type = "HOLD" if v['is_hold'] else f"TAP (CD: {v['cooldown']}s)"
                formatted_map[k] = f"{v['keys']} | {m_type} | Req: {v['hold_time']}s"
            else:
                formatted_map[k] = v # Fallback for old legacy saves
                
        self.text_view.display_dataset(unique_gestures, self.y_labels, formatted_map)

    def cmd_new_gesture(self):
        self.cmd_list_gestures()
        self.text_view.show_message("\n--- NEW GESTURE ---")
        new_name = self.text_view.get_input("Enter name for new gesture (or press Enter to cancel):")
        
        if not new_name:
            self.text_view.show_message("Operation cancelled.\n---")
            return
            
        self.state['current_training_gesture'] = new_name
        macro_key = self.text_view.get_input("Type key/combination (e.g., 'alt+f4') or press Enter for no macro:")
        
        if macro_key:
            # --- NEW ATTRIBUTE PROMPTS ---
            is_hold_input = self.text_view.get_input("Is this a continuous HOLD macro? (y/n):").lower()
            is_hold = (is_hold_input == 'y')
            
            # Ask for required activation time (default to 0.1)
            hold_req_str = self.text_view.get_input("Hold time required to trigger (seconds) [Default 0.1]:")
            hold_req = float(hold_req_str) if hold_req_str else 0.1
            
            cooldown = 0.0
            if not is_hold:
                # Only ask for cooldown if it's a tap macro
                cd_str = self.text_view.get_input("Cooldown between activations (seconds) [Default 1.0]:")
                cooldown = float(cd_str) if cd_str else 1.0
                
            # Save as a dictionary instead of a string
            self.macro_map[self.state['current_training_gesture']] = {
                'keys': macro_key,
                'is_hold': is_hold,
                'hold_time': hold_req,
                'cooldown': cooldown
            }
            self.text_view.show_message(f"Assigned '{macro_key}' with custom attributes.")
            # -----------------------------
        else:
            self.text_view.show_message("No macro assigned. Gesture will only be tracked visually.")
            
        self.text_view.show_message(f"\n[!] READY TO RECORD: '{self.state['current_training_gesture']}'")
        self.text_view.show_message(">>> Click back to the video window and HOLD the SPACEBAR to capture frames.")
        
        unique_classes = set(self.y_labels)
        unique_classes.add(self.state['current_training_gesture'])
        if len(unique_classes) < 2:
            self.text_view.show_message("\n>>> HINT: You need at least 2 distinct gestures to train the model.")
        self.text_view.show_message("---\n")

    def cmd_record_frame(self, latest_result):
        if self.state['current_training_gesture'] == "None":
            return

        if not latest_result or not latest_result.hand_landmarks:
            return

        for hand_landmarks in latest_result.hand_landmarks:
            flat_data = normalize_landmarks(hand_landmarks)
            self.X_data.append(flat_data)
            self.y_labels.append(self.state['current_training_gesture'])
        
        count = self.y_labels.count(self.state['current_training_gesture'])
        self.state['action_msg'] = f"RECORDING... ({count} frames)"
        
        if count % 10 == 0:
            self.text_view.show_message(f"-> Recording '{self.state['current_training_gesture']}': {count} frames captured...")

    def cmd_training_status(self):
        unique_gestures = set(self.y_labels)
        self.text_view.display_training_status(self.state['current_training_gesture'], unique_gestures, self.y_labels)
    
    def cmd_modify_macro(self):
        self.cmd_list_gestures()
        self.text_view.show_message("\n--- UPDATE MACRO ---")
        target = self.text_view.get_input("Enter gesture to update (or press Enter to cancel):")
        
        if not target:
            self.text_view.show_message("Operation cancelled.\n---")
            return
            
        if target in self.y_labels:
            macro_key = self.text_view.get_input("Type new key/combination (or press Enter to remove macro):")
            
            if macro_key:
                # --- NEW ATTRIBUTE PROMPTS ---
                is_hold_input = self.text_view.get_input("Is this a continuous HOLD macro? (y/n):").lower()
                is_hold = (is_hold_input == 'y')
                
                hold_req_str = self.text_view.get_input("Hold time required to trigger (seconds) [Default 0.1]:")
                hold_req = float(hold_req_str) if hold_req_str else 0.1
                
                cooldown = 0.0
                if not is_hold:
                    cd_str = self.text_view.get_input("Cooldown between activations (seconds) [Default 1.0]:")
                    cooldown = float(cd_str) if cd_str else 1.0
                    
                self.macro_map[target] = {
                    'keys': macro_key,
                    'is_hold': is_hold,
                    'hold_time': hold_req,
                    'cooldown': cooldown
                }
                self.text_view.show_message(f"Success! Updated '{target}' with '{macro_key}' and custom attributes.")
                # -----------------------------
            else:
                if target in self.macro_map:
                    del self.macro_map[target]
                self.text_view.show_message(f"Macro removed. Gesture '{target}' will only be tracked visually.")
                
            save_dataset(self.X_data, self.y_labels, self.macro_map)
        else:
            self.text_view.show_message(f"Gesture '{target}' not found in dataset.\n---")
            
        self.text_view.show_message("---\n")

    def cmd_remove_gesture(self):
        self.cmd_list_gestures()
        self.text_view.show_message("\n--- REMOVE GESTURE ---")
        
        unique_gestures = set(self.y_labels)
        if not unique_gestures:
            self.text_view.show_message("No gestures recorded yet to remove.\n---")
            return

        # 1. Update the prompt to hint at the 'ALL' command
        target = self.text_view.get_input("Gesture to remove (or type 'ALL' to wipe everything, Enter to cancel):")
        
        if not target:
            self.text_view.show_message("Operation cancelled.\n---")
            return
            
        # 2. Add the 'ALL' intercept block
        if target.lower() == 'all':
            self.text_view.show_message("\n[CRITICAL WARNING] You are about to delete EVERY gesture and wipe the AI brain.")
            confirm = self.text_view.get_input("Type 'yes' to permanently delete everything:").lower()
            
            if confirm == 'yes':
                # Clear all active memory
                self.X_data.clear()
                self.y_labels.clear()
                self.macro_map.clear()
                self.custom_classifier = None
                self.state['current_training_gesture'] = "None"
                
                # Delete the physical files
                if os.path.exists('models/gesture_dataset.pkl'):
                    os.remove('models/gesture_dataset.pkl')
                if os.path.exists('models/my_custom_gestures.pkl'):
                    os.remove('models/my_custom_gestures.pkl')
                    
                self.text_view.show_message("-> SUCCESS: All gestures and models have been completely wiped.")
            else:
                self.text_view.show_message("Aborted. Your gestures are safe.\n---")
            return

        # 3. Standard single-gesture removal logic continues below...
        if target not in self.y_labels:
            self.text_view.show_message(f"Gesture '{target}' not found in dataset.\n---")
            return
            
        if len(unique_gestures) <= 2:
            self.text_view.show_message(f"\n[WARNING] You currently have {len(unique_gestures)} distinct gestures.")
            self.text_view.show_message("Removing one will leave you with fewer than 2, meaning the model can no longer function.")
            self.text_view.show_message("This will reset your trained AI model.")
            confirm = self.text_view.get_input("Are you sure you want to proceed? (yes/no):").lower()
            if confirm != 'yes':
                self.text_view.show_message("Aborted.\n---")
                return

        self.X_data, self.y_labels = remove_gesture(self.X_data, self.y_labels, target)
        if target in self.macro_map: 
            del self.macro_map[target]
            
        if len(set(self.y_labels)) < 2:
            self.custom_classifier = None
            if os.path.exists('models/my_custom_gestures.pkl'):
                os.remove('models/my_custom_gestures.pkl')
            self.text_view.show_message("-> Active model wiped from memory due to insufficient gestures.")
        else:
            self.text_view.show_message("-> Auto-retraining model to clear removed gesture from memory...")
            save_dataset(self.X_data, self.y_labels, self.macro_map)
            self.custom_classifier = train_custom_gestures(self.X_data, self.y_labels)
            self.text_view.show_message(f"Successfully removed '{target}'.")
        self.text_view.show_message("---\n")

    def cmd_train_model(self):
        unique_classes = set(self.y_labels)
        if len(unique_classes) < 2:
            self.text_view.show_message("Cannot train! You need at least 2 distinct gestures.")
            self.text_view.show_message(f"You currently have {len(unique_classes)}: {list(unique_classes)}")
            self.state['current_training_gesture'] = "None"
        else:
            save_dataset(self.X_data, self.y_labels, self.macro_map)
            self.custom_classifier = train_custom_gestures(self.X_data, self.y_labels)
            self.state['current_training_gesture'] = "None"