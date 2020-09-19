from os.path import abspath, join

import cv2
import numpy as np

from graph import create_graph
from draw import OverlayDrawer
from rectify import ImageRectifier


DATA_PATH = join(abspath('.'), 'data')


def start_capture():
    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)
    return cap


def get_click_event_handler(capture):
    def capture_still(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("click detected")
            capture.read()

    return capture_still


if __name__ == "__main__":
    # TODO: Replace mock graph with graph data from real detection
    graph = create_graph(
        nodes=[
            {'id': "G1", 'label': "G", 'coordinates': [128, 250]},
            {'id': "Ts1", 'label': "Ts", 'coordinates': [250, 150]},
            {'id': "P1", 'label': "Ps", 'coordinates': [401, 151]},
            {'id': "C1", 'label': "C", 'coordinates': [403, 250]},
            {'id': "Fs1", 'label': "Fs", 'coordinates': [250, 350]}
        ],
        edges=[
            {'source': "G1", 'target': "Ts1"},
            {'source': "Ts1", 'target': "P1"},
            {'source': "P1", 'target': "C1"},
            {'source': "C1", 'target': "Fs1"},
            {'source': "Fs1", 'target': "G1"},
            {'source': "C1", 'target': "G1"}
        ]
    )
    drawer = OverlayDrawer(graph)

    capture = start_capture()
    cv2.namedWindow('window')
    cv2.setMouseCallback('window', get_click_event_handler(capture))
    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()

        # Blank frame
        (width, height) = frame.shape[:2]

        # Rectify captured image
        #image_rectifier = ImageRectifier(frame)
        #image_rectifier = ImageRectifier(join(DATA_PATH, 'examples_png/ex1_pic.png'))
        #cv2.imshow('rectified', image_rectifier.img)

        # TODO: Detect symbols/edges and build graph
        # -> detect.py

        # Overlay the detected graph over original image
        drawn_image = drawer.draw(*frame.shape[:2])

        # FIXME: More complicated adding needed so the overlay is not transparent
        # Where blank_image values are not 0, we want to override the values in gray
        gray = cv2.addWeighted(drawn_image, 1.0, frame, 1.0, 0.0)
        #gray = gray + drawn_image

        # Display the resulting composed image
        cv2.imshow('window', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
