from time import sleep

import cv2
import imutils
from imutils.video import VideoStream


class Camera:
    _default_image_path = f"../classes/ready.jfif"

    def __init__(self, res_w=800, res_h=800):
        self.resolution = [res_h, res_w]
        self.camera = VideoStream(src=0, resolution=(self.resolution[0], self.resolution[1])).start()
        sleep(2.0)

        self.Update_Monitor(self.Read())
        """
        _default_img = imutils.resize(
            cv2.imread(self._default_image_path),
            width=self.resolution[0],
            height=self.resolution[1]
        )
        self.Update_Monitor(_default_img)
        """


    def Read(self):
        return imutils.resize(
            self.camera.read(),
            width=self.resolution[0],
            height=self.resolution[1]
        )


    def Update_Monitor(self, img):
        cv2.imshow("Frame", img)
        cv2.waitKey(1) & 0xFF
