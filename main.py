import cv2
import mediapipe
import pyautogui

camera = cv2.VideoCapture(0)
while True: 
    _,image = camera.read()
    cv2.imshow("Eye controlled mouse", image)
    key = cv2.waitKey(100)
    if key == 27: 
        break
camera.release()
cv2.destroyAllWindows()

