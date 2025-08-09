import cv2
import mediapipe
import pyautogui

face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
camera = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
while True: 
    _,image = camera.read()
    image = cv2.flip(image,flipCode=1)
    window_height,window_width,_ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmark_points = processed_image.multi_face_landmarks

    if all_face_landmark_points: 
        one_face_landmark_points = all_face_landmark_points[0].landmark

        right_eye = one_face_landmark_points[474:478] #474-477 (478 not included) are right eye points
        for id,landmark_point in enumerate(right_eye):
            x = int(landmark_point.x * window_width)
            y = int(landmark_point.y * window_height)
            #print(x, y)

            if id == 1: # if face exists
                mouse_x = int((screen_width / window_width) * x)
                mouse_y = int((screen_height / window_height) * y)
                pyautogui.moveTo(mouse_x, mouse_y)
            
            cv2.circle(image, (x,y), radius=3, color=(0, 0, 255))
            
        left_eye = [one_face_landmark_points[145], one_face_landmark_points[159]] #145 and 159 are left eye points
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_width)
            y = int(landmark_point.y * window_height)
            #print(x, y)
            cv2.circle(image, (x,y), radius=3, color=(0, 255, 255))
        # if left eye is closed
        if (left_eye[0].y - left_eye[1].y < 0.01):
            pyautogui.click()
            pyautogui.sleep(2)
            print("mouse clicked")

    cv2.imshow("Eye controlled mouse", image)
    
    key = cv2.waitKey(100)
    if key == 27: # Escape key
        break
camera.release()
cv2.destroyAllWindows()

