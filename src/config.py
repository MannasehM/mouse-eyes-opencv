import numpy as np
import pyautogui

# Screen size
SCREEN_W, SCREEN_H = pyautogui.size()

# 3D model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Mediapipe landmark indices of 2D image that match the MODEL_POINTS order
LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Blink detection
EAR_THRESHOLD = 0.25       # Eye Aspect Ratio threshold
BLINK_HOLD_TIME = 0.50     # Minimum blink duration (seconds)

# Mouse sensitivity
SENSITIVITY_DEFAULT = 2