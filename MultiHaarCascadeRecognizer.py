import cv2
from classes.camera import Camera
from classes.HaarCascade_Model import Recognizer

ia = Recognizer(
    [
        {"name": "eye", "color": (255, 0, 0), "path": "./models/haarcascade_eye.xml"},
        #{"name": "upperbody", "color": (0, 255, 0), "path": "./models/haarcascade_upperbody.xml"},
        #{"name": "bottle", "color": (0, 0, 255), "path": "./models/bottleV2.xml"}
    ]
)

webcam = Camera()


while True:
    frame = webcam.Read()
    detections = ia.Check(frame)

    if len(detections) != 0:
        for model_detection in detections:
            for detection_index in range(len(model_detection.get("frame_detection").detection_result)):

                x       = model_detection.get("frame_detection").detection_result[detection_index][0]
                y       = model_detection.get("frame_detection").detection_result[detection_index][1]
                width   = model_detection.get("frame_detection").detection_result[detection_index][2]
                height  = model_detection.get("frame_detection").detection_result[detection_index][3]

                cv2.rectangle(
                        frame,
                        (x, y),
                        (x + height, y + width),
                        model_detection.get("color"),
                        5
                    )

                cv2.putText(
                        frame,
                        f"{model_detection.get('name')}: {100-model_detection.get('frame_detection').rejectLevels[detection_index]}%",
                        (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                )

    webcam.Update_Monitor(frame)
