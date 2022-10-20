import cv2

from classes.camera import Camera
from classes.model_recognizer import Recognizer

ia = Recognizer("./models/haarcascade_eye.xml")

webcam = Camera()

while True:
    frame = webcam.Read()
    found = ia.Detect(frame)
    print(f"Founds: {found}")
    print()
    if len(found) != 0:
        for (x, y, width, height) in found:

            cv2.rectangle(frame,
                          (x, y),
                          (x + height, y + width),
                          (0, 255, 0),
                          5)

            cv2.putText(frame,
                        "STOP",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2)

    webcam.Update_Monitor(frame)
