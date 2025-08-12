import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import math
from collections import deque

# DO NOT CHANGE
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# DO NOT CHANGE
LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# data from mediapipe
face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Camera internal parameters
camera = cv2.VideoCapture(0)
_,image = camera.read()
size = image.shape
print(size)
focal_length = size[1] #width
center = (size[1] / 2, size[0] / 2) 

# denoted as K and uses camera internal params for projecting 3d scene onto 2d image
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

# no distortion (in camera lens) assumed
dist_coeffs = np.zeros((4,1)) 

# Screen parameters
screen_width, screen_height = pyautogui.size()

# function for EAR to see how open an eye is
def eye_aspect_ratio(eye):
    A = math.dist((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    B = math.dist((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    C = math.dist((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    return (A + B) / (2 * C)

# function to convert rotation matrix to euler angles (roll, pitch, yaw)
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])  # Roll
        y = math.atan2(-R[2,0], sy)      # Pitch
        z = math.atan2(R[1,0], R[0,0])   # Yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # Convert from radians to degrees
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])


# Settings
SMOOTHING_WINDOW = 5  # Bigger = smoother but more delay
HORIZ_SPEED = 2       # Adjust sensitivity
VERT_SPEED = 1.5

# Calibration variables
is_calibrated = False
calib_yaw = 0
calib_pitch = 0

# For smoothing
yaw_history = deque(maxlen=SMOOTHING_WINDOW)
pitch_history = deque(maxlen=SMOOTHING_WINDOW)

num_of_iterations = 0
while True: 
    _,image = camera.read()
    image = cv2.flip(image,flipCode=1)
    window_height,window_width,_ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmark_points = processed_image.multi_face_landmarks

    if all_face_landmark_points: 
        one_face_landmark_points = all_face_landmark_points[0].landmark
        key_landmark_indices = []

        for id in LANDMARK_IDS:
            landmark = one_face_landmark_points[id]
            x = int(landmark.x * size[1])
            y = int(landmark.y * size[0])
            key_landmark_indices.append((x, y))
            cv2.circle(image, (x, y), 3, (0, 0, 255))

        # solvePnP requires nparray array argument
        key_landmark_indices = np.array(key_landmark_indices, dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, key_landmark_indices, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            nose_end_point3D = np.array([[0, 0, 1000.0]])
            nose_end_point2D, _ = cv2.projectPoints(
                nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs
            )

            p1 = (int(key_landmark_indices[0][0]), int(key_landmark_indices[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(image, p1, p2, (255, 0, 0), 2)

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            euler_angles = rotationMatrixToEulerAngles(rotation_matrix)
            roll, pitch, yaw = euler_angles # roll not needed

            if not is_calibrated: 
                cv2.putText(image, "Hold head still - Press 'c' to calibrate", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else: 
                yaw -= calib_yaw
                pitch -= calib_pitch

            # Define sensitivity or scaling factors (tweak experimentally)
            # sensitivity_x = 20  # degrees to pixels ratio (example)
            # sensitivity_y = 20

            # Map yaw and pitch to screen coordinates
            # center_x = screen_width / 2
            # center_y = screen_height / 2
            yaw_history.append(yaw)
            pitch_history.append(pitch)
            smoothed_yaw   = np.mean(yaw_history)
            smoothed_pitch = np.mean(pitch_history)
            # mouse_x = center_x + smoothed_pitch * sensitivity_x
            # mouse_y = center_y - smoothed_yaw * sensitivity_y

            # Clamp mouse_x and mouse_y to screen bounds
            # mouse_x = max(0, min(screen_width - 1, mouse_x))
            # mouse_y = max(0, min(screen_height - 1, mouse_y))

            print(f"smoothed_yaw: {smoothed_yaw}")
            # print(f"smoothed_pitch: {smoothed_pitch}")
            # print(f"mouse_x: {mouse_x}")
            # print(f"mouse_y: {mouse_y}")

            # Move mouse
            move_x = smoothed_yaw * HORIZ_SPEED
            move_y = -smoothed_pitch * VERT_SPEED # negative so nodding down moves cursor down
            pyautogui.moveRel(move_x, move_y) 
        
        # right_eye = one_face_landmark_points[474:478] #474-477 (478 not included) are right eye points
        # for id,landmark_point in enumerate(right_eye):
        #     x = int(landmark_point.x * window_width)
        #     y = int(landmark_point.y * window_height)
        #     #print(x, y)
        #     if id == 1: # if face exists
        #         mouse_x = int((screen_width / window_width) * x)
        #         mouse_y = int((screen_height / window_height) * y)
        #         #pyautogui.moveTo(mouse_x, mouse_y)
        #     cv2.circle(image, (x,y), radius=3, color=(0, 0, 255))

        left_eye = [one_face_landmark_points[33], one_face_landmark_points[160], one_face_landmark_points[158], one_face_landmark_points[133], one_face_landmark_points[153], one_face_landmark_points[144]]
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_width)
            y = int(landmark_point.y * window_height)
            cv2.circle(image, (x,y), radius=3, color=(0, 255, 128))

        if num_of_iterations == 0: 
            is_closed = False

        left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
        # left eye is closed
        if left_eye_aspect_ratio < 0.25: 
            if is_closed == False: 
                closed_start_time = time.time()
            is_closed = True
        else: 
            if is_closed == True: 
                closed_end_time = time.time()
                blink_duration = closed_end_time - closed_start_time
                #print(blink_duration)

                if blink_duration > 0.50: 
                    #pyautogui.click()
                    pyautogui.sleep(1)
                    print("mouse clicked")

            is_closed = False

    cv2.imshow("Eye/head controlled mouse", image)
    
    key = cv2.waitKey(100)

    # Press 'c' to calibrate
    if key == ord('c'):
        calib_yaw = yaw
        calib_pitch = pitch
        calibration_done = True
        print(f"Calibrated: yaw={calib_yaw:.2f}, pitch={calib_pitch:.2f}")

    if key == 27: # Escape key
        print("Pressed escape key. Closed windows.")
        break

    num_of_iterations = num_of_iterations + 1
camera.release()
cv2.destroyAllWindows()