# Import the necessary packages
import imutils  # pip install imutils
import dlib  # pip install dlib
import cv2  # pip install opencv-python

# To get the current time
from datetime import datetime

# for efficiency of time testing
start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Starting Time =", start_time)
###########################################################

# Initialize the image path
image_path = "images/face.jpeg"
image = cv2.imread(image_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize dlib's face detector (CNN-based)
d = "models/mmod_human_face_detector.dat"
detector = dlib.cnn_face_detection_model_v1(d)

# Detect faces in the image
rectangles = detector(gray, 1)

# Loop over the face detections
for face in rectangles:
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
end_time = end.strftime("%H:%M:%S")
print("End Time =", end_time)
duration = end - start
duration_in_s = duration.total_seconds()
print(duration_in_s)
##################################################

cv2.waitKey(0)
