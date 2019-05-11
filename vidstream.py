from imutils.video import VideoStream
import cv2
import numpy as np
import time
import picamera

vs = VideoStream(usePiCamera = True, resolution=(480, 360)).start()
time.sleep(2.0)


while True :
    image = vs.read()
    print(image.shape)
    cv2.imshow('live',image)

    key = cv2.waitKey(1) & 0xFF
    if key==ord('q') :
        break

cv2.destroyAllWindows()
vs.stop()

