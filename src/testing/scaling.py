import cv2

cap = cv2.VideoCapture(0)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    rect, frame = cap.read()
    frame75 = rescale_frame(frame, percent=75)
    cv2.imshow('frame75', frame75)
    frame100 = rescale_frame(frame, percent=100)
    cv2.imshow('frame100', frame100)
    
    key = cv2.waitKey(100)
    if key == 27: # Escape key
        break

cap.release()
cv2.destroyAllWindows()