import cv2

class FrameDetection:
    def __init__(self, model, frame, scale=1.1, minNeighbors=6, minSize=(30,30), getConfidence=1):
        self.detection_result, self.rejectLevels, self.levelWeights = model.detectMultiScale3(frame, scaleFactor=scale, minNeighbors=minNeighbors, minSize=minSize, outputRejectLevels=getConfidence)


class Model:
    def __init__(self, model_info: dict, width_min_scale_analyse, height_min_scale_analyse):
        self.name = model_info.get("name", "error no name")
        self.model = cv2.CascadeClassifier(model_info.get("path", "error no path"))
        self.color = model_info.get("color", (200, 200, 200))
        self.min_scale_analyse = [width_min_scale_analyse, height_min_scale_analyse]


    def Detect(self, img) -> FrameDetection:
        """
        3 methode to get dection
        detectMultiScale:
            return self.model.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                minSize=(self.min_scale_analyse[0], self.min_scale_analyse[1])
        )
        detectMultiScale2
        detectMultiScale3:
             object = self.model.detectMultiScale3(
                gray,
               scaleFactor = 1.1,
               minNeighbors = 5,
               minSize = (30, 30),
               flags = cv2.CASCADE_SCALE_IMAGE,
               outputRejectLevels = True
            )
            return {
                "zone": object[0]
                "neighbours": object[1]
                "weights": object[2]
            }
        """
        return FrameDetection(self.model, img, minSize=(self.min_scale_analyse[0], self.min_scale_analyse[1]))


class Recognizer:
    def __init__(self, models: list, width_min_scale_analyse=50, height_min_scale_analyse=50):
        # init models
        self.models = [Model(model, width_min_scale_analyse, height_min_scale_analyse) for model in models]

    def Check(self, frame):
        founds = []
        for model in self.models:
            founds.append({"name": model.name, "color": model.color, "frame_detection": model.Detect(frame)})

        return founds