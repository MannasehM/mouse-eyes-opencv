import cv2
import mediapipe as mp

#face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # better nose/eye/mouth accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
camera = cv2.VideoCapture(0)

while True: 
    _,image = camera.read()
    image = cv2.flip(image,flipCode=1)

    window_height,window_width,_ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmark_points = processed_image.multi_face_landmarks

    if all_face_landmark_points: 
        one_face_landmark_points = all_face_landmark_points[0].landmark

        for id,landmark_point in enumerate(one_face_landmark_points):
            x = int(landmark_point.x * window_width)
            y = int(landmark_point.y * window_height)
            cv2.circle(image, (x,y), radius=3, color=(0, 255, 0))
            cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.imshow("Landmark indices", image)
    
    key = cv2.waitKey(100)
    if key == 27: # Escape key
        break
camera.release()
cv2.destroyAllWindows()

