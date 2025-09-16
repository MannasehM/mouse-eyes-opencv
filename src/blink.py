import time
import pyautogui
import math

class BlinkDetector:
    def __init__(self, ear_threshold, hold_time):
        self.ear_threshold = ear_threshold
        self.hold_time = hold_time
        self.is_closed = False
        self.closed_start_time = 0

    def eye_aspect_ratio(self, eye):
        A = math.dist((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
        B = math.dist((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
        C = math.dist((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
        return (A + B) / (2 * C)

    def detect_blink(self, eye):
        ear = self.eye_aspect_ratio(eye)
        if ear < self.ear_threshold:
            if not self.is_closed:
                self.closed_start_time = time.time()
            self.is_closed = True
        else:
            if self.is_closed:
                duration = time.time() - self.closed_start_time
                if duration > self.hold_time:
                    pyautogui.click()
                    print("Mouse clicked")
                    time.sleep(0.5)
            self.is_closed = False