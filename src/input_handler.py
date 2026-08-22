try:
    from evdev import UInput, ecodes as e
    virtual_keyboard = UInput()
except PermissionError:
    print("FATAL: Permission denied for virtual keyboard. Try running with sudo.")
    exit()

EDGE_CASES = {
    "ctrl": e.KEY_LEFTCTRL, "alt": e.KEY_LEFTALT, 
    "shift": e.KEY_LEFTSHIFT, "win": e.KEY_LEFTMETA, 
    "windows": e.KEY_LEFTMETA, "page up": e.KEY_PAGEUP, 
    "page down": e.KEY_PAGEDOWN, "caps lock": e.KEY_CAPSLOCK, 
    "volumemute": e.KEY_MUTE,
    "up": e.KEY_UP, "down": e.KEY_DOWN,
    "left": e.KEY_LEFT, "right": e.KEY_RIGHT
}

class InputHandler:
    def __init__(self):
        self.held_keys = set() # Track what is currently pressed

    def _get_valid_codes(self, key_string):
        """Helper method to parse the string into evdev codes."""
        keys_to_press = key_string.split('+')
        valid_codes = []
        for k in keys_to_press:
            k = k.strip().lower()
            code = EDGE_CASES.get(k)
            if not code:
                target_key_name = f"KEY_{k.upper()}"
                if hasattr(e, target_key_name):
                    code = getattr(e, target_key_name)
            if code:
                valid_codes.append(code)
        return valid_codes

    def press_macro(self, key_string):
        """Presses keys down and holds them."""
        codes = self._get_valid_codes(key_string)
        for code in codes:
            if code not in self.held_keys:
                virtual_keyboard.write(e.EV_KEY, code, 1)
                self.held_keys.add(code)
        virtual_keyboard.syn()

    def release_macro(self, key_string):
        """Releases specific held keys."""
        codes = self._get_valid_codes(key_string)
        for code in reversed(codes):
            if code in self.held_keys:
                virtual_keyboard.write(e.EV_KEY, code, 0)
                self.held_keys.remove(code)
        virtual_keyboard.syn()

    def tap_macro(self, key_string):
        """Standard behavior: Presses and immediately releases keys."""
        self.press_macro(key_string)
        self.release_macro(key_string)
        
    def release_all(self):
        """Failsafe to release everything when switching modes or gestures."""
        for code in list(self.held_keys):
            virtual_keyboard.write(e.EV_KEY, code, 0)
        self.held_keys.clear()
        virtual_keyboard.syn()

    def close(self):
        self.release_all()
        virtual_keyboard.close()