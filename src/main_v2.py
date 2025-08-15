import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import math
from collections import deque

# -------------------------------------------------------CONSTANTS-------------------------------------------------------

# 3D model points of key facial landmarks (for head pose estimation)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Mediapipe landmark indices of 2D image that match the model_points order
LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Blink detection parameters
EAR_THRESHOLD = 0.25       # Eye Aspect Ratio threshold
BLINK_HOLD_TIME = 0.50     # Minimum blink duration (seconds)

# Smoothing for head movement
SMOOTHING_WINDOW = 10  # Bigger = smoother but more lag

# Sensitivity for head movement (tweak experimentally)
SENSITIVITY_X = 30 # degrees to pixels ratio
SENSITIVITY_Y = 15

# Screen params
SCREEN_W, SCREEN_H = pyautogui.size()

# -------------------------------------------------------STATE VARIABLES-------------------------------------------------------

# Blink detection state
is_closed = False
closed_start_time = 0

# Calibration state
is_calibrated = False
calib_yaw = 0
calib_pitch = 0

# Smoothing history buffers
yaw_history = deque(maxlen=SMOOTHING_WINDOW)
pitch_history = deque(maxlen=SMOOTHING_WINDOW)

# -------------------------------------------------------INIT-------------------------------------------------------

# Initialize mediapipe face mesh
face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# OpenCV camera setup
camera = cv2.VideoCapture(0)
_, image = camera.read()
size = image.shape  # (height, width, channels)
print(f"Camera resolution: {size}")

# Camera internal params
focal_length = size[1] # width
center = (size[1] / 2, size[0] / 2) 
camera_matrix = np.array([ # denoted as K and uses camera internal params for projecting 3d scene onto 2d image
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4,1)) # assuming no lens distortion

# -------------------------------------------------------FUNCTIONS-------------------------------------------------------

def eye_aspect_ratio(eye):
    A = math.dist((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    B = math.dist((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    C = math.dist((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    return (A + B) / (2 * C)

def get_all_face_landmark_points(image, face_mesh_landmarks):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmark_points = processed_image.multi_face_landmarks
    return all_face_landmark_points

def detect_blink(eye):
    global is_closed, closed_start_time
    ear = eye_aspect_ratio(eye)

    # if eye is closed
    if ear < EAR_THRESHOLD: 
        if is_closed == False: 
            closed_start_time = time.time()
        is_closed = True
    else: 
        if is_closed == True: 
            closed_end_time = time.time()
            blink_duration = closed_end_time - closed_start_time
            print(blink_duration)

            if blink_duration > BLINK_HOLD_TIME: 
                pyautogui.click()
                pyautogui.sleep(1)
                print("mouse clicked")
        is_closed = False

def rotationMatrixToEulerAngles(R):
    # function to convert rotation matrix to euler angles (roll, pitch, yaw)
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

def smooth_angles(yaw, pitch):
    yaw_history.append(yaw)
    pitch_history.append(pitch)
    return np.mean(yaw_history), np.mean(pitch_history)

def move_mouse(smoothed_yaw, smoothed_pitch):
    screen_center_x = SCREEN_W / 2
    screen_center_y = SCREEN_H / 2
    mouse_x = screen_center_x + smoothed_yaw * SENSITIVITY_X
    mouse_y = screen_center_y - smoothed_pitch * SENSITIVITY_Y
    mouse_x = max(0, min(SCREEN_W - 1, mouse_x))
    mouse_y = max(0, min(SCREEN_H - 1, mouse_y))
    # pyautogui.moveTo(mouse_x, mouse_y)

    # print(f"smoothed_yaw: {smoothed_yaw}")
    # print(f"smoothed_pitch: {smoothed_pitch}")
    # print(f"mouse_x: {mouse_x}")
    # print(f"mouse_y: {mouse_y}")

    # move_x = smoothed_yaw * HORIZ_SPEED
    # move_y = -smoothed_pitch * VERT_SPEED # negative so nodding down moves cursor down
    # pyautogui.moveRel(move_x, move_y) 

def draw_nose_line(rotation_vector, translation_vector, camera_matrix, dist_coeffs, image, points_2D):
    nose_end_point3D = np.array([[0, 0, 1000.0]])
    nose_end_point2D, _ = cv2.projectPoints(
        nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    p1 = (int(points_2D[0][0]), int(points_2D[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(image, p1, p2, (255, 0, 0), 2)

def get_head_pose(one_face_landmark_points, image, model_points, camera_matrix, dist_coeffs):
    points_2D = []
    for id in LANDMARK_IDS:
        landmark = one_face_landmark_points[id]
        x = int(landmark.x * size[1])
        y = int(landmark.y * size[0])
        points_2D.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 0, 255))

    # solvePnP requires nparray argument
    points_2D = np.array(points_2D, dtype="double")

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, points_2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success: 
        return None, None, None, None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    roll, pitch, yaw = rotationMatrixToEulerAngles(rotation_matrix) # roll not needed

    return rotation_vector, translation_vector, points_2D, roll, pitch, yaw

# -------------------------------------------------------MAIN-------------------------------------------------------

while True: 
    _,image = camera.read()
    image = cv2.flip(image,flipCode=1)
    window_height,window_width,_ = image.shape
    all_face_landmark_points = all_face_landmark_points = get_all_face_landmark_points(image, face_mesh_landmarks)

    if all_face_landmark_points: 
        one_face_landmark_points = all_face_landmark_points[0].landmark
        
        rotation_vector, translation_vector, points_2D, roll, pitch, yaw = get_head_pose(one_face_landmark_points, image, model_points, camera_matrix, dist_coeffs)
        if yaw is not None:
            draw_nose_line(rotation_vector, translation_vector, camera_matrix, dist_coeffs, image, points_2D)
            if not is_calibrated: 
                cv2.putText(image, "Hold head still - Press 'c' to calibrate", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:  
                yaw -= calib_yaw
                pitch -= calib_pitch
                smoothed_yaw, smoothed_pitch = smooth_angles(yaw, pitch)
                move_mouse(smoothed_yaw, smoothed_pitch)

        # Left eye landmarks
        left_eye = [one_face_landmark_points[33], one_face_landmark_points[160], one_face_landmark_points[158], one_face_landmark_points[133], one_face_landmark_points[153], one_face_landmark_points[144]]
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_width)
            y = int(landmark_point.y * window_height)
            cv2.circle(image, (x,y), radius=3, color=(0, 255, 128))
        detect_blink(left_eye)

    cv2.imshow("Eye/head controlled mouse", image)
    key = cv2.waitKey(100)

    # Press 'c' to calibrate
    if key == ord('c'):
        calib_yaw = yaw
        calib_pitch = pitch
        is_calibrated = True
        print(f"Calibrated: yaw={calib_yaw:.2f}, pitch={calib_pitch:.2f}")
    if key == 27: # Escape key
        print("Pressed escape key. Closed windows.")
        break
camera.release()
cv2.destroyAllWindows()