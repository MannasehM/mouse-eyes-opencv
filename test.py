import cv2
import mediapipe as mp
import numpy as np
import math

face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

camera = cv2.VideoCapture(0)

ret, image = camera.read()
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

while True:
    _, image = camera.read()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_landmarks.process(rgb_image)

    if results.multi_face_landmarks:
        one_face_landmarks = results.multi_face_landmarks[0].landmark
        image_points = []

        for idx in LANDMARK_IDS:
            lm = one_face_landmarks[idx]
            x = int(lm.x * size[1])
            y = int(lm.y * size[0])
            image_points.append((x, y))
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # solvePnP requires nparray array argument
        image_points = np.array(image_points, dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            nose_end_point3D = np.array([[0, 0, 1000.0]])
            nose_end_point2D, _ = cv2.projectPoints(
                nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs
            )

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(image, p1, p2, (255, 0, 0), 2)

    cv2.imshow("Head Pose Estimation", image)

    key = cv2.waitKey(100)
    if key == 27: # Escape key
        break
camera.release()
cv2.destroyAllWindows()