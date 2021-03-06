# import the necessary packages
import dlib  # pip install dlib
import cv2  # pip install opencv-python

# To get the current time
from datetime import datetime

# For efficiency of time testing
start = datetime.now()
print("Starting Time =", start)
###########################################################

# initialize dlib's face detector (CNN-based)
d = "models/mmod_human_face_detector.dat"
detector = dlib.cnn_face_detection_model_v1(d)

cap = cv2.VideoCapture(0)

while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for face in rects:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "CNN-Based", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    # show the output image with the face detections
    cv2.imshow("Facial Detection Dlib CNN Based", image)

    ##################################################
    end = datetime.now()
    print("End Time =", end)
    duration = end - start
    duration_in_s = duration.total_seconds()
    print(duration_in_s)
    ##################################################

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
