import argparse
import sys

import cv2
import numpy as np


class Recognizer:
    def __init__(self, model_path, min_scale=[20, 20]):
        self.model = cv2.CascadeClassifier(model_path)
        self.min_scale_analyse = min_scale[:]

    def Detect(self, img):
        return self.model.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            minSize=(self.min_scale_analyse[0], self.min_scale_analyse[1])
        )


class MultiRecognizer:
    def __init__(self, model_path, protocol_path):
        def _set_args():
            if len(sys.argv) == 1:
                args = {
                    "prototxt": "MobileNetSSD_deploy.prototxt.txt",
                    "model": "MobileNetSSD_deploy.caffemodel",
                    "confidence": 0.2,
                }
            else:
                # lancement à partir du terminal
                # python3 ObjectRecognition.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
                ap = argparse.ArgumentParser()
                ap.add_argument("-p", "--prototxt", required=True,
                                help="path to Caffe 'deploy' prototxt file")
                ap.add_argument("-m", "--model", required=True,
                                help="path to Caffe pre-trained model")
                ap.add_argument("-c", "--confidence", type=float, default=0.2,
                                help="minimum probability to filter weak detections")
                args = vars(ap.parse_args())

            return args

        self.args = _set_args()
        self.object_to_recognize = ["arriere-plan", "avion", "velo", "oiseau", "bateau",
                                    "bouteille", "autobus", "voiture", "chat", "chaise", "vache", "table",
                                    "chien", "cheval", "moto", "personne", "plante en pot", "mouton",
                                    "sofa", "train", "moniteur"]

        self.object_colors = np.random.uniform(0, 255, size=(len(self.object_to_recognize), 3))
        self.model = cv2.dnn.readNetFromCaffe(protocol_path, model_path)

    def Detect(self, img):
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        # Feed input to neural network
        self.model.setInput(blob)
        detections = self.model.forward()
        return self.model.forward()
