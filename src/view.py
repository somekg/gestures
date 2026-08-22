import cv2
from src.vision import draw_landmarks_on_image

class GestureUI:
    """Handles drawing the OpenCV video window."""
    def render(self, frame, state, latest_result):
        # Draw skeleton using the dynamic state setting
        if latest_result is not None:
            frame = draw_landmarks_on_image(
                frame, 
                latest_result, 
                allowed_hands=state.get('allowed_hands', ["Left", "Right"])
            )

        # Draw Header
        macro_text = "Macros: ON (Live)" if state['macros_enabled'] else "Macros: PAUSED"
        color = (0, 0, 255) if state['macros_enabled'] else (150, 150, 150)
        cv2.putText(frame, f"{macro_text} - Press 'e' toggle", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Target: '{state['current_training_gesture']}'", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw dynamic messages (predictions, charging state, errors)
        if state.get('overlay_msg'):
            cv2.putText(frame, state['overlay_msg'], (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, state.get('msg_color', (0, 255, 0)), 2)
            
        if state.get('action_msg'):
            cv2.putText(frame, state['action_msg'], (20, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow('MediaPipe Gesture Recognition', frame)
        return cv2.waitKey(1) & 0xFF


class TerminalView:
    """Handles all terminal-based text output and input."""
    def show_message(self, message):
        print(message)

    def get_input(self, prompt):
        return input(f"Terminal Input -> {prompt} ").strip()

    def display_dataset(self, unique_gestures, y_labels, macro_map):
        print("\n=====================================================")
        print("                CURRENT DATASET")
        print("=====================================================")
        if not unique_gestures:
            print("No gestures recorded yet.")
        else:
            for gesture in unique_gestures:
                frame_count = y_labels.count(gesture)
                macro = macro_map.get(gesture, "None (Visual Only)")
                print(f"- {gesture} ({frame_count} frames) -> Macro: {macro}")
                
            print(f"\nTotal Gestures: {len(unique_gestures)}")
            print(f"Total Frames: {len(y_labels)}")
        print("=====================================================\n")

    def display_training_status(self, current_target, unique_gestures, y_labels):
        print("\n=====================================================")
        print("             CURRENT TRAINING STATUS")
        print("=====================================================")
        print(f"Active Target for SPACEBAR: '{current_target}'")
        print(f"\nDistinct Gestures in Memory: {len(unique_gestures)} / 2 required to train")
        
        # New block: List the gestures currently in the dataset
        if unique_gestures:
            print("Gestures queued for training:")
            for i, gesture in enumerate(sorted(list(unique_gestures))):
                count = y_labels.count(gesture)
                print(f"  {i+1}. {gesture} ({count} frames)")
        else:
            print("Gestures queued for training: None")
        
        if len(unique_gestures) >= 2:
            print("\nStatus: [ READY TO TRAIN ] -> Press 't' to compile model.")
        else:
            print("\nStatus: [ NOT READY ] -> You must record more distinct gestures first.")
        print("=====================================================\n")


class GUIView:
    """Placeholder for future graphical buttons and popups."""
    def show_message(self, message):
        pass # To be implemented with a visual popup

    def get_input(self, prompt):
        pass # To be implemented with a text entry window

    def display_dataset(self, unique_gestures, y_labels, macro_map):
        pass # To be implemented with a listbox or table

    def display_training_status(self, current_target, unique_gestures):
        pass # To be implemented with status labels