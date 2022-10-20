import cv2
from matplotlib import pyplot as plt
from ImageLoader import LoadImage


img = LoadImage("a.jfif")
# Use minSize because for not
# bothering with extra-small
# dots that would look like STOP signs
stop_data = cv2.CascadeClassifier('./models/stop_data.xml')

found = stop_data.detectMultiScale(img["img_gray"], minSize=(20, 20))

# Don't do anything if there's
# no sign
amount_found = len(found)

if amount_found != 0:

    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:

        # We draw a green rectangle around
        # every recognized sign
        cv2.rectangle(img["img_rgb"], (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 5)

# Creates the environment of
# the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img["img_rgb"])
plt.show()