import cv2
import mediapipe as mp
from pynput import keyboard
import pyautogui
from blink import BlinkDetector
from head_pose import HeadPose
import config
import numpy as np

class HeadMouseController:
    """
    Main controller for combining head pose and blink detection
    to move the mouse and simulate clicks.
    """
    def __init__(self):
        # --------------------- CAMERA SETUP ---------------------
        self.camera = cv2.VideoCapture(0)
        _, frame = self.camera.read()
        self.size = frame.shape

        print("------------------------------------------------------")
        print("Head/Eye Mouse Control Started!")
        print("Instructions:")
        print(" - Move your head to control the mouse cursor.")
        print(" - Blink (hold) your left eye to click.")
        print(" - Press 'h' to toggle head control on/off.")
        print(" - Use the slider to adjust sensitivity.")
        print(" - Press ESC to exit.")
        print("------------------------------------------------------")

        focal_length = self.size[1]
        center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))  # assume no distortion

        # --------------------- FACE MESH ---------------------
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --------------------- BLINK DETECTOR ---------------------
        self.blink_detector = BlinkDetector(config.EAR_THRESHOLD, config.BLINK_HOLD_TIME)

        # --------------------- HEAD POSE ---------------------
        self.head_pose = HeadPose(config.MODEL_POINTS, config.LANDMARK_IDS, self.camera_matrix, self.dist_coeffs)
        
        self.head_control_enabled = True
        self.sensitivity = config.SENSITIVITY_DEFAULT

        # --------------------- KEYBOARD LISTENER ---------------------
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.window_name = "Head/Eye Mouse Control"
        cv2.namedWindow(self.window_name)
        self.max_sensitivity = 5  # Max sensitivity allowed
        cv2.createTrackbar("Sensitivity", self.window_name, int(self.sensitivity*10), int(self.max_sensitivity*10), self.update_sensitivity)

    def on_press(self, key):
        try:
            if key.char == 'h': # toggle head control
                self.head_control_enabled = not self.head_control_enabled
                print(f"Head control {'enabled' if self.head_control_enabled else 'disabled'}")
        except AttributeError:
            pass # ignore special keys

    def update_sensitivity(self, val):
        clamped_val = min(val, int(self.max_sensitivity * 10))  # Clamp to max
        self.sensitivity = clamped_val / 10

    def move_mouse(self, p1, p2):
        """
        Map nose line direction to screen coordinates and move the mouse.
        """
        if self.head_control_enabled:
            dx = (p2[0]-p1[0])*self.sensitivity
            dy = (p2[1]-p1[1])*self.sensitivity
            pyautogui.moveTo(config.SCREEN_W/2 + dx, config.SCREEN_H/2 + dy)

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret: continue
            frame = cv2.flip(frame, 1) # mirror for natural movement
            faces = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            if faces:
                landmarks = faces[0].landmark

                # HEAD POSE
                rvec, tvec, points_2D = self.head_pose.get_head_pose(landmarks, frame)
                if rvec is not None:
                    p1, p2 = self.head_pose.draw_nose_line(rvec, tvec, frame, points_2D)
                    self.move_mouse(p1, p2)

                # LEFT EYE FOR BLINK
                left_eye = [landmarks[33], landmarks[160], landmarks[158],
                            landmarks[133], landmarks[153], landmarks[144]]
                for lm in left_eye:
                    x, y = int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])
                    cv2.circle(frame, (x,y), 3, (0,255,128))
                self.blink_detector.detect_blink(left_eye)

            cv2.imshow(self.window_name, frame)

            # ESC key to exit
            if cv2.waitKey(10) == 27:
                print("Pressed escape key. Closed windows.")
                break

        self.camera.release()
        cv2.destroyAllWindows()

