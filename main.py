import cv2
import mediapipe
import pyautogui
import time
import math

face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
camera = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

def eye_aspect_ratio(eye):
    A = math.dist((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    B = math.dist((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    C = math.dist((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    return (A + B) / (2 * C)


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

        right_eye = one_face_landmark_points[474:478] #474-477 (478 not included) are right eye points
        for id,landmark_point in enumerate(right_eye):
            x = int(landmark_point.x * window_width)
            y = int(landmark_point.y * window_height)
            #print(x, y)
            if id == 1: # if face exists
                mouse_x = int((screen_width / window_width) * x)
                mouse_y = int((screen_height / window_height) * y)
                #pyautogui.moveTo(mouse_x, mouse_y)
            cv2.circle(image, (x,y), radius=3, color=(0, 0, 255))

        left_eye = [one_face_landmark_points[33], one_face_landmark_points[160], one_face_landmark_points[158], one_face_landmark_points[133], one_face_landmark_points[153], one_face_landmark_points[144]]
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_width)
            y = int(landmark_point.y * window_height)
            #print(x, y)
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
                print(blink_duration)

                if blink_duration > 0.50: 
                    pyautogui.click()
                    pyautogui.sleep(1)
                    print("mouse clicked")

            is_closed = False

    cv2.imshow("Eye controlled mouse", image)
    
    key = cv2.waitKey(100)
    if key == 27: # Escape key
        break

    num_of_iterations = num_of_iterations + 1
camera.release()
cv2.destroyAllWindows()

