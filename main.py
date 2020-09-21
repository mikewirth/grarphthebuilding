from os.path import abspath, join
import time

import cv2
import numpy as np

from graph import create_graph
from draw import OverlayDrawer
from detect import GraphGenerator
from rectify import ImageRectifier


DATA_PATH = join(abspath('.'), 'data')

IMAGE_FILENAME = 'ex0_big.png'
IMAGE_FILENAME = 'ex4.png'

# IMAGE_FILENAME = 'ex3.png'

APP_WINDOW = 'GrARph The Building - ' + IMAGE_FILENAME

AR_MODE = False

_global_wait_lock = True


def start_capture():
    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)
    return cap


if __name__ == "__main__":
    if AR_MODE:
        capture = start_capture()
        cv2.namedWindow('window')

        # Show video feed until diagram correctly positioned
        while(True):
            ret, frame = capture.read()
            cv2.imshow('window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # Start real detection loop
    while(True):
        # Capture frame-by-frame
        if AR_MODE:
            ret, frame = capture.read()
        else:
            frame = cv2.imread(join(DATA_PATH, "examples_png/" + IMAGE_FILENAME))

        # Blank frame
        (height, width) = frame.shape[:2]

        # Rectify captured image
        print("Trying to rectify the image...")
        t_start = time.time()
        image_rectifier = ImageRectifier(frame)
        #image_rectifier = ImageRectifier(join(DATA_PATH, 'examples_png/ex1_pic.png'))
        #cv2.imshow('rectified', image_rectifier.img)

        # TODO: Detect symbols/edges and build graph
        #detected_graph = GraphGenerator(cv2.imread(join(DATA_PATH, 'examples_png/ex0.png'))).graph
        print("Rectified in %f seconds... now detecting the graph" % (time.time() - t_start))
        detected_graph = GraphGenerator(frame).graph

        # Overlay the detected graph over original image
        print("Graph detected in %f seconds! Now overlaying the symbols..." % (time.time() - t_start))
        graph = create_graph(**detected_graph)
        overlay_image = OverlayDrawer(graph).draw(*frame.shape[:2])

        # FIXME: More complicated adding needed so the overlay is not transparent
        # Where blank_image values are not 0, we want to override the values in gray
        gray = cv2.addWeighted(overlay_image, 1.0, frame, 0.5, 0.5)
        #gray = gray + overlay_image

        # Display the resulting composed image
        cv2.namedWindow(APP_WINDOW, cv2.WINDOW_NORMAL)
        # gray = cv2.resize(gray, (1024, int(1024 * (height/width))))
        cv2.imshow(APP_WINDOW, gray)

        if AR_MODE:
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break
        else:        
            cv2.waitKey(0) # & 0xFF in [27, ord('q')]:
            break

    if AR_MODE:
        # When everything done, release the capture
        capture.release()

    cv2.destroyAllWindows()
