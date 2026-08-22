# AI Hand Gesture Macro Controller

A modular, real-time Python application that uses computer vision and machine learning to translate custom hand gestures into system-wide keyboard macros. 

This app allows users to easily train their own custom hand gestures on the fly, assign configurable Tap or Hold macros, and execute them universally across any Linux application.

## ✨ Key Features
*   **Real-Time Custom Training:** Record and train new gestures directly through the app in seconds without touching any code.
*   **Smart AI Gatekeeper:** Utilizes a `OneClassSVM` gatekeeper paired with a `CalibratedClassifierCV` to confidently reject unknown hand positions (dead zones) and prevent accidental macro fires.
*   **Advanced Macro Logic:** Supports both single-fire "Tap" macros with configurable cool-downs, and continuous "Hold" macros with required activation times.
*   **Universal Execution:** Uses the Linux `evdev` library to simulate hardware-level keyboard inputs, ensuring macros work seamlessly in video games, browsers, and text editors.
*   **Visual Overlay:** Built-in OpenCV window displaying live skeleton tracking, dynamic confidence scores, and macro activation statuses.

## 🛠️ Prerequisites
*   **Operating System:** Linux strictly (Required for `evdev` virtual keyboard injection).
*   **Python:** Python 3.10-3.12 (When making the project MediaPipe did not support python 3.13 or higher)
*   **Hardware:** A standard webcam.
*   **Permissions:** Executing virtual keyboard inputs via `/dev/uinput` requires root access by default in Linux, you'll have to grant specific access to your user, there's a guide at the end for this.

## 🚀 Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME
```
**2. Setup the python environment:**
```bash
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```
**3. Download the official Mediapipe task:**
> Note: If the wget download fails, you can manually download the current gesture_recognizer.task model from the Official MediaPipe Documentation.
```bash
wget -O gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

### 🔐 Running Without Sudo (Recommended Setup)
> Quick tip: Apps like Steam already do something like this, you can try running the program directly or checking the permissions of /dev/input for a + at the end (file has an Access Control List)

Follow these steps to permanently set up secure, sudo-free access:

#### Step 1: Create a Custom `udev` Rule
```bash
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
``` 
#### Step 2: Add Your User to the `input` Group
```bash
sudo usermod -aG input $USER
```
#### Step 3: Reload Linux Permissions
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
#### Step 4: Restart computer
You can now activate your environment and run the gesture application safely without sudo:

## 🏃 Running command
```bash
python main.py
```
