from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import requests

def send_task(task, skuid):
    url = "http://127.0.0.1:5000/task"
    headers = {"Content-Type": "application/json"}
    data = {"task": task, "skuid": str(skuid)}
    response = requests.post(url, headers=headers, json=data)
    print(f"Sent {task} task for SKUID {skuid}: {response.text}")

# Load the YOLOv8 model
model = YOLO("yolov8n-face.pt")

# Open the video file
cap = cv2.VideoCapture(0)

# Store the track history
track_history = defaultdict(lambda: [])
active_tracks = set()
all_ids = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        current_frame_tracks = set()

        if results[0].boxes and results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            all_ids = track_ids

            for track_id in track_ids:
                current_frame_tracks.add(track_id)

            new_tracks = current_frame_tracks - active_tracks
            lost_tracks = active_tracks - current_frame_tracks

            for track_id in new_tracks:
                send_task("start", track_id)

            for track_id in lost_tracks:
                send_task("stop", track_id)

            active_tracks = current_frame_tracks

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 20:  # retain 20 tracks for 20 frames
                    track.pop(0)

                # Draw the tracking lines
                if len(track) > 1:  # Ensure there are enough points to draw
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            # If no boxes are detected, just show the current frame
            cv2.imshow("YOLOv8 Tracking", frame)
            for id in all_ids:
                send_task("stop", id)
                all_ids = []

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

