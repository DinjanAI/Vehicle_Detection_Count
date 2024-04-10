import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone

cap = cv2.VideoCapture('D:/counting vehicles/vehicle detection and counting coures/test3.mp4')
model = YOLO('D:/counting vehicles/vehicle detection and counting coures/yolov8n.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

tracker = Sort(max_age=20)

frame_width = 1090  # Assuming the frame width is 1050 pixels
enter_line = [300, 400, 650, 400]  
exit_line = [700, 500, frame_width, 500]   

enter_counter = []
exit_counter = []

while 1:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('D:/counting vehicles/vehicle detection and counting coures/test3.mp4')
        continue
    detections = np.empty((0, 5))
    result = model(frame, stream=1)
    for info in result:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            objectdetect = classnames[classindex]

            if objectdetect in ['car', 'bus', 'truck'] and conf > 10:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

    track_result = tracker.update(detections)
    cv2.line(frame, (enter_line[0], enter_line[1]), (enter_line[2], enter_line[3]), (255, 0, 0), 4)
    cv2.line(frame, (exit_line[0], exit_line[1]), (exit_line[2], exit_line[3]), (255, 0, 255), 4)

    for results in track_result:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cvzone.putTextRect(frame, f'{id}',
                           [x1 + 8, y1 - 12], thickness=2, scale=1.5)

        # Count entering vehicles
        if enter_line[0] < cx < enter_line[2] and enter_line[1] - 20 < cy < enter_line[1] + 20:
            if id not in enter_counter:
                enter_counter.append(id)

        # Count leaving vehicles
        if exit_line[0] < cx < exit_line[2] and exit_line[1] - 20 < cy < exit_line[1] + 20:
            if id not in exit_counter:
                exit_counter.append(id)

    cvzone.putTextRect(frame, f'Entering Vehicles = {len(enter_counter)}', [290, 34], thickness=4, scale=2.3, border=2)
    cvzone.putTextRect(frame, f'Leaving Vehicles = {len(exit_counter)}', [290, 80], thickness=4, scale=2.3, border=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(33) == 27: 
        break
    cv2.waitKey(1)
