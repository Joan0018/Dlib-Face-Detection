# Import the necessary packages
from imutils import face_utils  # pip install imutils
import imutils
import dlib  # pip install dlib
import cv2  # pip install opencv-python

# To get the current time
from datetime import datetime

# For efficiency of time testing
start = datetime.now()
print("Starting Time =", start)
###########################################################

# Initialize the image path
image_path = "images/face.jpeg"
image = cv2.imread(image_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize dlib's face detector (HOG-based)
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()

# Detect faces in the image
rectangles = detector(gray, 1)


# loop over the face detections
for rect in rectangles:

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

cv2.waitKey(0)

