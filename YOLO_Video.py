from ultralytics import YOLO
import cv2
import math
import cvzone


def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
#     *'mp4v'), 10, (frame_width, frame_height))

    model = YOLO(model="./best.pt")
    classNames = ['cheetah', 'elephant', 'lion', 'rhino']
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
    #             # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                if cls == 0:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC= (139, 69, 19))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1)
                elif cls == 1:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC= (139, 69, 19), colorR=(255,0,0))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0,0,0))
                elif cls == 2:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC= (139, 69, 19), colorR=(0,255,0))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0,0,0))
                elif cls == 3:
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC= (139, 69, 19), colorR=(0,0,255))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                        0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0,0,0))
        yield img
        # out.write(img)
        # cv2.imshow("image", img)
        # if cv2.waitKey(1) & 0xFF==ord('1'):
        # break
    # out.release()
