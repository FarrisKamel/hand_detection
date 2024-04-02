import cv2
import mediapipe as mp
import numpy as np
import uuid     # Used to generate Unique identifier
import os       # OS library for python

mp_drawing = mp.solutions.drawing_utils     # Easier for rendering landmarks
mp_hands = mp.solutions.hands               

cap = cv2.VideoCapture(0)       # Get the webcam feed

# Instantiating the mp hand model
# Detection confidence = 0.8
# Tracking confidence = 0.5
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

    # Read through each frame
    while cap.isOpened():
        ret, frame = cap.read()     #return value, frame varaible

        # Recolor the frame  CV.BGR -> mp.RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set the neccessary flags
        image.flags.writeable = False
        
        # Detections 
        results = hands.process(image)
        
        # Set drawing flag to true 
        image.flags.writable = True
        
        # Recolor the frame  mp.RGB -> CV.BGR
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # See the detection
        print(results)

        cv2.imshow('Hand Tracking', frame)      #Render image to screen

        user_input = cv2.waitKey(10)    # Save user input
        
        # Quit if q or ESC are clicked 
        if user_input == ord('q') or user_input == 27:
            break

# Used to close all operations smoothly
cap.release()
cv2.destroyAllWindows()



