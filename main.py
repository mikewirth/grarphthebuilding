import cv2
import numpy as np

from image_rectifier import ImageRectifier
from graph_generator import GraphGenerator


def main():
    image_rectifier = ImageRectifier('data/examples_png/ex0_pic.png')
    rectified_img = image_rectifier.img
    graph_generator = GraphGenerator(rectified_img)
    print(graph_generator.graph)


if __name__ == "__main__":
    main()
