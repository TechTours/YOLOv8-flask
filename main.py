import os
import cv2
import math
from fastapi import FastAPI, UploadFile
import requests
import datetime
from typing import Optional
import tempfile
from ultralytics import YOLO
import numpy as np
import random

app = FastAPI()

# Define the latitude and longitude ranges
lowest_latitude = -2.316111
highest_latitude = -1.308633
lowest_longitude = 30.593512
highest_longitude = 30.825589

# Define the path where the video will be saved
output_video_path = "output_video.mp4"
classNames = ['Cheetah', 'Elephant', 'Lion', 'Rhino']
detected_animals = []

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, 30, (640, 480))  # Adjust frame size as needed

# API endpoint to send detected animal information
API_URL = "http://194.163.167.131:8020/activity/create"

class DetectedAnimalLocation:
    latitude: Optional[float]
    longitude: Optional[float]

# Initialize variables to keep track of detected animals
detected_animals = {}  # Dictionary to store {animal_id: (last_detected_frame, detected_animal)}
next_animal_id = 1  # Initialize the next animal ID to 1

def save_and_send_image(frame, latitude, longitude, detected_animal):
    # Send data to the API
    timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
    data = {
        "animal": detected_animal,
        "location": "Akagera National Park",  # Update with the actual location
        "time": timestamp,
        "longitude": str(longitude),
        "latitude": str(latitude)
    }

    response = requests.post(API_URL, json=data)

    if response.status_code == 201:
        print(f"Image saved and data sent to API for {detected_animal} at ({latitude}, {longitude}).")
    else:
        print(f"Failed to send data to API. Status code: {response.status_code}")

def generate_frames(video_file):
    global detected_animals 
    global next_animal_id
    cap = cv2.VideoCapture(video_file)

    # Initialize YOLO model
    model = YOLO(model="./best.pt")

    while True:
        success, img = cap.read()
        if not success:
            break

        # Reset the animal counts for each frame
        animal_counts = {cls: 0 for cls in classNames}

        # Perform YOLO detection on the frame
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                if cls >= 0 and cls < len(classNames):
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # Update the animal counts for the detected class
                    animal_counts[classNames[cls]] += 1

                    # If an animal is detected with confidence >= 0.7 and is not already tracked, assign an ID
                    if conf >= 0.9 and detected_animals.get(cls) is None:
                        latitude, longitude = generate_random_coordinates()
                        detected_animal = classNames[cls]
                        animal_id = next_animal_id
                        next_animal_id += 1  # Increment the next animal ID
                        detected_animals[animal_id] = (cap.get(cv2.CAP_PROP_POS_FRAMES), detected_animal)
                        save_and_send_image(img, latitude, longitude, detected_animal)

        # Remove animals that have not been detected in the last few frames (e.g., 10 frames)
        frames_to_keep = cap.get(cv2.CAP_PROP_POS_FRAMES)
        detected_animals = {animal_id: (frame, animal) for animal_id, (frame, animal) in detected_animals.items()
                            if frames_to_keep - frame < 10}

        # Add text to display animal counts in the current frame
        text = 'Animal Counts in Frame: ' + ', '.join([f'{cls}: {count}' for cls, count in animal_counts.items()])
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        yield cv2.imencode('.jpg', img)[1].tobytes()

def generate_random_coordinates():
    # Generate random latitude and longitude within the specified ranges
    latitude = random.uniform(lowest_latitude, highest_latitude)
    longitude = random.uniform(lowest_longitude, highest_longitude)
    return latitude, longitude

@app.post("/process_video/")
async def process_video(video_file: UploadFile):
    global detected_animals 
    # Save the uploaded video file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_path = temp_video_file.name
        temp_video_file.write(video_file.file.read())

    # Call the generate_frames function to process the uploaded video
    for frame_data in generate_frames(temp_video_path):
        # Save the frame to the output video
        output_video.write(np.frombuffer(frame_data, dtype=np.uint8))

    # Release the output video
    output_video.release()

    # Remove the temporary video file
    os.remove(temp_video_path)

    return {"message": "Video processed and images saved."}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Animal Detection API!"}
