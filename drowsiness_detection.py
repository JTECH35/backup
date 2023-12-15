# Import the necessary packages
import datetime as dt
from EAR_calculator import *
from imutils import face_utils
from imutils.video import VideoStream
from matplotlib import style
import imutils
import dlib
import time
import argparse
import cv2
import os
import pyttsx3
from threading import Thread
from ultralytics import YOLO
import cvzone
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from playsound import playsound
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import matplotlib.animation as animate




# Initialize text-to-speech engine
text_speech = pyttsx3.init()

# Function to trigger an alarm with a given message
def alarm(msg):
    text_speech.say(msg)
    text_speech.runAndWait()
    time.sleep(0.5)


alarm_status = False
alarm_status2 = False
saying = False

style.use('fivethirtyeight')


# Creating the dataset
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


# Lists to store eye and mouth aspect ratios along with timestamps
ear_list = []
total_ear = []
mar_list = []
total_mar = []
ts = []
total_ts = []




# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether raspberry pi camera shall be used or not")
args = vars(ap.parse_args())


# Constants for eye and mouth aspect ratios, blink frames, and MAR threshold
EAR_THRESHOLD = 0.27
CONSECUTIVE_FRAMES = 10
MAR_THRESHOLD = 25

# Initialize counters
TOTAL_BLINKS = 0
BLINK_COUNT = 0
FRAME_COUNT = 0
CLOSED_EYES_FRAME = 3

# Load face detector and predictor models
print("[INFO]Loading the predictor.....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Indices for facial landmarks
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video stream setup
print("[INFO]Loading Camera.....")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
vs.stream.set(cv2.CAP_PROP_FPS, 60)

# Ensure dataset directory exists
assure_path_exists("dataset/")
count_sleep = 0
count_yawn = 0

# YOLO model initialization
model = YOLO("../Yolo-Weights/best (3).pt")
classNames = ["awake","drowsy"]



# Main loop for video processing
while True:
    # Extract a frame
    frame = vs.read()
    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # YOLO object detection
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Adjust the y-coordinate to create space between the text and bounding box
            text_y_position = max(35, y1 - 2)

            # Define color based on class
            if currentClass == 'awake':
                myColor = (0, 255, 0)
            else:
                myColor = (0, 0, 255)

            # Display class and confidence
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), text_y_position), scale=1, thickness=1,
                               colorB=myColor,
                               colorT=(255, 255, 255), colorR=myColor, offset=0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)

    # Detect faces using dlib
    rects = detector(blurred, 1)

    # Process each detected face
    for (i, rect) in enumerate(rects):
        shape = predictor(blurred, rect)
        # Convert it to a (68, 2) size numpy array
        shape = face_utils.shape_to_np(shape)


        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]
        mouth = shape[mstart:mend]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0

        # Record eye aspect ratio and timestamp
        ear_list.append(EAR)
        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        # Compute the convex hull for both the eyes and then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        # Draw facial landmarks(circles)
        for (x, y) in shape[lstart:lend]:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        for (x, y) in shape[rstart:rend]:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        for (x, y) in shape[mstart:mend]:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Compute and record mouth aspect ratio
        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR / 10)



        # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place
        # Thus, count the number of frames for which the eye remains closed
        if EAR < EAR_THRESHOLD:
            FRAME_COUNT += 1
            BLINK_COUNT += 1
            # Draw facial contours
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            # If consecutive closed eyes frames exceed the threshold, trigger an alert
            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                count_sleep += 1
                if alarm_status == False:
                    # Set the alarm status to True to avoid repeated triggering
                    alarm_status = True
                    t = Thread(target=alarm, args=('please wake up',))
                    t.deamon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # If eyes are not closed, reset the consecutive frame count and alarm status
            FRAME_COUNT = 0
            alarm_status = False

            # Check if the number of blinks exceeds a certain threshold
            if BLINK_COUNT >= CLOSED_EYES_FRAME:
                # Increment the total blink count
                TOTAL_BLINKS += 1
            # Reset the blink count for the current sequence
            BLINK_COUNT = 0



#### COMMENTED ####
            # Display total blink count
          #  cv2.putText(frame, "Total Blinks: {}".format(TOTAL_BLINKS), (255, 90),
                    #    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#### COMMENTED ####

        # Check for yawning
        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            #Trigger yawn alert
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('please take some fresh air',))
                t.deamon = True
                t.start()
        else:
            alarm_status2 = False

    # display the frame
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit loop if 'q' is pressed
    if key == ord('q'):
        cv2.destroyWindow("Output")
        break
# Release video stream
vs.stop()