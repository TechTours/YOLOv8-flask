from flask import Flask, render_template, Response, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2
import math
import cvzone
from ultralytics import YOLO

app = Flask(__name__)

app.config['SECRET_KEY'] = 'laurent'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['IMAGE_FOLDER'] = 'static/images'  # New folder to store captured images

# Create a counter for image filenames
image_counter = 0
classNames = ['cheetah', 'elephant', 'lion', 'rhino']
animal_counts = {cls: 0 for cls in classNames}  # Counter for each animal class

# Use FlaskForm to get input video file from the user
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x=''):
    global image_counter

    cap = cv2.VideoCapture(path_x)
    
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
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 255), t=3)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
                    
                    # Update the animal counts for the detected class
                    animal_counts[classNames[cls]] += 1

        # Add text to display animal counts in the current frame
        text = 'Animal Counts in Frame: ' + ', '.join([f'{cls}: {count}' for cls, count in animal_counts.items()])
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img)[1].tobytes() + b'\r\n')

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

            return latitude, longitude
        else:
            return None

def add_location_to_image(image_path, latitude, longitude):
    # Open the image for writing
    with open(image_path, 'rb') as file:
        img = cv2.imread(image_path)
        # Create an ExifTag for latitude and longitude
        latitude_ref = 'S' if latitude < 0 else 'N'
        longitude_ref = 'W' if longitude < 0 else 'E'
        lat_deg = abs(int(latitude))
        lat_min = (abs(latitude) - lat_deg) * 60
        lon_deg = abs(int(longitude))
        lon_min = (abs(longitude) - lon_deg) * 60
        tags = {
            'GPS GPSLatitude': f"{lat_deg}/1,{lat_min}/1,0/1",
            'GPS GPSLatitudeRef': latitude_ref,
            'GPS GPSLongitude': f"{lon_deg}/1,{lon_min}/1,0/1",
            'GPS GPSLongitudeRef': longitude_ref,
        }
        # Add the GPS tags to the image's EXIF metadata
        exif_bytes = exifread.dump_tags(tags)
        file.seek(0)
        file.write(exif_bytes)

# Rest of your code ...

# Inside the function where you save the image:
def save_image(frame, filename, latitude, longitude):
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['IMAGE_FOLDER'])
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, filename)
    cv2.imwrite(image_path, frame)
    # Add location information to the image's metadata
    add_location_to_image(image_path, latitude, longitude)

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    return render_template('video.html', form=form)

@app.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
