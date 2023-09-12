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
# current_class = None

# Use FlaskForm to get input video file from the user
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x=''):
    global image_counter
    global current_class

    current_class = None
    cap = cv2.VideoCapture(path_x)
    
    # Initialize YOLO model
    model = YOLO(model="./best.pt")
    
    while True:
        success, img = cap.read()
        if not success:
            break

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
                if cls == 0:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1)
                elif cls == 1:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19), colorR=(255, 0, 0))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
                elif cls == 2:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19), colorR=(0, 255, 0))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
                elif cls == 3:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=(139, 69, 19), colorR=(0, 0, 255))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
                    # Check if the class is different from the current class
                if cls != current_class:
                    # Take a picture
                    img_name = f'Detected {classNames[cls]}_{image_counter}.jpg'
                    save_image(img, img_name)

                    # Update the current class
                    current_class = cls

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img)[1].tobytes() + b'\r\n')
        image_counter += 1

def save_image(frame, filename):
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['IMAGE_FOLDER'])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), 'wb') as img_file:
        img_file.write(cv2.imencode('.jpg', frame)[1])

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
