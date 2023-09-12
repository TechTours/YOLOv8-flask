from flask import Flask, request, jsonify, Response, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2
import math
import cvzone
from ultralytics import YOLO
import exifread  # Import exifread for GPS data extraction

app = Flask(__name__)

app.config['SECRET_KEY'] = 'laurent'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['IMAGE_FOLDER'] = 'static/images'

# Global variables
image_counter = 0
classNames = ['cheetah', 'elephant', 'lion', 'rhino']

# Initialize YOLO model
yolo_model = YOLO(model="./best.pt")

# Function to save an image
def save_image(frame, filename):
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['IMAGE_FOLDER'])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), 'wb') as img_file:
        img_file.write(cv2.imencode('.jpg', frame)[1])

# Function to extract GPS location info from an image
def extract_location_info(image_path):
    with open(image_path, 'rb') as file:
        tags = exifread.process_file(file)

        if 'GPS GPSLongitude' in tags and 'GPS GPSLatitude' in tags:
            latitude = tags['GPS GPSLatitude'].values
            longitude = tags['GPS GPSLongitude'].values

            latitude = float(latitude[0].num) / float(latitude[0].den)
            longitude = float(longitude[0].num) / float(longitude[0].den)

            if tags['GPS GPSLatitudeRef'].values == 'S':
                latitude = -latitude
            if tags['GPS GPSLongitudeRef'].values == 'W':
                longitude = -longitude

            return {'latitude': latitude, 'longitude': longitude}
        else:
            return None

# Define API endpoint for starting video processing
@app.route('/start_processing', methods=['POST'])
def start_processing():
    # Handle incoming requests, start video processing, and return a response.
    # You may want to pass parameters in the request data to control the processing.
    # Example: video_path = request.json.get('video_path')
    global image_counter

    # Start video processing...
    video_path = request.json.get('video_path')
    cap = cv2.VideoCapture(video_path)
    current_class = None

    def generate_frames():
        # nonlocal image_counter
        while True:
            success, img = cap.read()
            if not success:
                break

            # Perform YOLO detection on the frame
            results = yolo_model(img, stream=True)

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
                    if cls == 0:
                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19))
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                    elif cls == 1:
                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19), colorR=(255, 0, 0))
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
                    elif cls == 2:
                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19), colorR=(0, 255, 0))
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
                    elif cls == 3:
                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19), colorR=(0, 0, 255))
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))

                    # Check if the class is different from the current class
                    if cls != current_class:
                        # Take a picture
                        img_name = f'Detected {classNames[cls]}_{image_counter}.jpg'
                        save_image(img, img_name)

                        # Update the current class
                        current_class = cls

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            image_counter += 1

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define API endpoint for retrieving GPS location of an image
@app.route('/get_location/<image_name>', methods=['GET'])
def get_location(image_name):
    # Retrieve GPS location information for the specified image and return it as JSON.
    image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['IMAGE_FOLDER'], image_name)
    
    if os.path.exists(image_path):
        location = extract_location_info(image_path)
        if location:
            return jsonify(location)
        else:
            return jsonify({'message': 'No GPS information found in the image.'}), 404
    else:
        return jsonify({'message': 'Image not found'}), 404

if __name__ == "__main__":
    app.run(debug=True)
