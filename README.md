# Overview
This project was made during the beginning stages of my software development experience with IDenTV. 
Using background threading, this program allows user to send curl commands to the program to trigger a background thread that is used for hand detection.
The user can then send another curl command to kill the background thread. An skuid is needed for each thread.

## Prerequisite
Make sure that your python virtaul environment contains the following packages:
    
    mediapipe
    opencv-python

## Using the Program
After running the program, using another terminal window run the following to activate a background thread:

    curl -X POST http://127.0.0.1:5000/task -H "Content-Type: application/json" -d '{"task": "start", "skuid": "44334"}'

Running the following to kill the background thread:
    
    curl -X POST http://127.0.0.1:5000/task -H "Content-Type: application/json" -d '{"task": "stop", "skuid": "44334"}'


