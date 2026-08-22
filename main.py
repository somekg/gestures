from src.controller import AppController

def print_startup_guide():
    """Prints a clean cheat sheet of all available commands."""
    print("=====================================================")
    print("         GESTURE MACRO APP - INITIALIZED")
    print("=====================================================")
    print(" GLOBAL CONTROLS (Always Active):")
    print("   [ q ] : Save dataset and Quit")
    print("   [ e ] : Toggle Mode (Macros LIVE vs. Setup Mode)")
    print("")
    print(" SETUP CONTROLS (Only Active when Macros are PAUSED):")
    print("   [ n ] : Create a NEW gesture and assign a macro")
    print("   [Space] : HOLD to record frames for the active gesture")
    print("   [ m ] : MODIFY an existing gesture's macro assignment")
    print("   [ r ] : REMOVE a gesture from the dataset")
    print("   [ l ] : LIST all current gestures and assigned macros")
    print("   [ s ] : STATUS check for training requirements")
    print("   [ t ] : TRAIN the machine learning model")
    print("=====================================================\n")

if __name__ == "__main__":
    print("Initializing Gesture Macro App...")

    print_startup_guide()

    app = AppController()
    
    try:
        app.run()
    finally:
        # Guarantee the keyboard is safely closed if the app crashes
        app.input_handler.close()
        print("Clean exit.")