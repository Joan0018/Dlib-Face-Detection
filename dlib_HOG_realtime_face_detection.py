# import the necessary packages
from imutils import face_utils  # pip install imutils
import dlib  # pip install dlib
import cv2  # pip install opencv-python

# To get the current time
from datetime import datetime

# for efficiency of time testing
start = datetime.now()
print("Starting Time =", start)
###########################################################

# initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "HOG-Based", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    # show the output image with the face detections
    cv2.imshow("Facial Detection Dlib HOG Based ", image)

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
