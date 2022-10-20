from classes.camera import Camera
from classes.model_recognizer import MultiRecognizer
from time import sleep
import cv2

import numpy as np

ia = MultiRecognizer(model_path="./models/MobileNetSSD_deploy.caffemodel", protocol_path="./models/MobileNetSSD_deploy.prototxt.txt")
webcam = Camera(res_w=800, res_h=800)

while True:
    frame = webcam.Read()
    found = ia.Detect(frame)
    (h, w) = frame.shape[:2]
    for i in np.arange(0, found.shape[2]):
            # Compute Object detection probability
            precision = found[0, 0, i, 2]

            if precision > 0.2:
                # Get index and position of detected object
                idx = int(found[0, 0, i, 1])
                box = found[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Create box and label
                label = "{}: {:.2f}%".format(ia.object_to_recognize[idx],
                                             precision * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              ia.object_colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ia.object_colors[idx], 2)


    webcam.Update_Monitor(frame)
