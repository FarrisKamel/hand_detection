import logging
import queue
import threading
import time
from queue import Queue
from abc import abstractmethod, ABC
from typing import Dict
import cv2
import mediapipe as mp
import numpy as np
import uuid     # Used to generate Unique identifier
import os       # OS library for python
from time import sleep
from configparser import ConfigParser

mp_drawing = mp.solutions.drawing_utils     # Easier for rendering landmarks
mp_hands = mp.solutions.hands               

# extract all the data configs set up by .ini file
config = ConfigParser()
config.read("parser.ini")
data = config["DEFAULT"]
bluriness_threshold = data["image_blurriness_threshold"]

TASKS_QUEUE = Queue()

class BackgroundThread(threading.Thread, ABC):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    @abstractmethod
    def startup(self) -> None:
        """
        Method that is called before the thread starts.
        Initialize all necessary resources here.
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self) -> None:
        """
        Method that is called shortly after stop() method was called.
        Use it to clean up all resources before thread stops.
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def handle(self) -> None:
        """
        Method that should contain business logic of the thread.
        Will be executed in the loop until stop() method is called.
        Must not block for a long time.
        :return: None
        """
        raise NotImplementedError()

    def run(self) -> None:
        """
        This method will be executed in a separate thread
        when start() method is called.
        :return: None
        """
        self.startup()
        while not self._stopped():
            self.handle()
        self.shutdown()


class HandDetectionThread(BackgroundThread):
    def startup(self) -> None:
        logging.info('HandDetectionThread started')

    def shutdown(self) -> None:
        logging.info('HandDetectionThread stopped')

    def handle(self) -> None:
        try:
            data = TASKS_QUEUE.get(block=False)
            task = data["task"]
            skuid = data["skuid"]
            #self.send_notification(skuid)
            self.handDetection(task, skuid)
            logging.info(f'Notification for {task} was sent.')
        except queue.Empty:
            time.sleep(1)

    def send_notification(self, skuid):
        logging.info(f"Start: Email to {skuid} was sent")
    
    def send_stop_notification(self, skuid):
        logging.info(f"Stop: Email to {skuid} was sent")

    # Function used to determine if the image if blurry
    def isBlurry(self, image):
        threshold = int(bluriness_threshold)                        # Set a Threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)              # Convert to the color needed
        laplace_gray = cv2.Laplacian(gray, cv2.CV_64F).var()        # Preform laplace
        is_blurry = False                                           # Bool to see if image blurry

        # Check if the image if blurry based on the threshold given
        if laplace_gray < threshold:            
            is_blurry = True

        # Add description to the image (can be removed)
        if not is_blurry:
            cv2.putText(image, "{}: {:.2f}".format("Not Blurry", laplace_gray), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        return is_blurry, image
    
    def handDetection(self, task, skuid):
        cap = cv2.VideoCapture(0)
        #create a dictory to store the image of the id
        skuid_dir = os.path.join("images", str(skuid))
        if not os.path.exists(skuid_dir):
            os.makedirs(skuid_dir)

        try:
            with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
                # Read through each frame
                while cap.isOpened() and not self._stopped():
                    ret, frame = cap.read()  # return value, frame varaible
                    if not ret:
                        print("Error: No frame is being read")
                        break # If no frame is read, break processing

                    # Recolor the frame  CV.BGR -> mp.RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Set the neccessary flags
                    image.flags.writeable = False

                    # Detections
                    results = hands.process(image)
                    hand_detected = False               # Bool to see if hand was detected

                    # Set drawing flag to true
                    image.flags.writeable = True

                    # Recolor the frame  mp.RGB -> CV.BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Call isBlurry function to check if the image is blurry
                    is_blurry, image = self.isBlurry(image)

                    # Render resutls if not blurry
                    if not is_blurry:
                        # Rendering results
                        if results.multi_hand_landmarks:
                            hand_detected = True            # hand detected

                            # Draw outline on image
                            for num, hand in enumerate(results.multi_hand_landmarks):  
                                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

                    # Save our image if hand detected
                    if not hand_detected:
                        cv2.imwrite(os.path.join(skuid_dir, '{}.jpg'.format(uuid.uuid1())), image)

                    #cv2.imshow('Hand Tracking', image)      # Display Camera
                    user_input = cv2.waitKey(10)  # Save user input

                    # Quit if q or ESC are clicked
                    if user_input == ord('q') or user_input == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        break

                    sleep(1)
        finally:
            cap.release()
            cv2.destroyAllWindows()

class BackgroundThreadFactory:
    @staticmethod
    def create(thread_type: str) -> BackgroundThread:
        if thread_type == 'handDetection':
            return HandDetectionThread()

        # if thread_type == 'some_other_type':
        #     return SomeOtherThread()

        raise NotImplementedError('Specified thread type is not implemented.')
