from ultralytics import YOLO
import cv2
import math
import cvzone


def image_detection(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Create the YOLO model
    model = YOLO(model="./best.pt")
    classNames = ['cheetah', 'elephant', 'lion', 'rhino']

    # Perform object detection on the image
    results = model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
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
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 0), colorR=(255, 0, 0))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                    0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
            elif cls == 2:
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 0), colorR=(0, 255, 0))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                    0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))
            elif cls == 3:
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 0, 0), colorR=(0, 0, 255))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                    0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 0))

    return img
