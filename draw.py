import math

import cv2
import numpy as np


NODE_SIZE_RADIUS = 25
BORDER_THICKNESS = 3
FONT = font = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1

COLOR_CYCLE = 1


def draw_node(canvas, node, color):
    node_x, node_y = node.coordinates

    textsize = cv2.getTextSize(node.label, FONT, FONT_SCALE, BORDER_THICKNESS)[0]
    text_x = (node_x - textsize[0]/2)
    text_y = (node_y + textsize[1]/2)

    cv2.putText(canvas, node.label, (int(text_x), int(text_y)), FONT, FONT_SCALE, color, BORDER_THICKNESS, cv2.LINE_AA)
    cv2.circle(canvas, node.coordinates, NODE_SIZE_RADIUS, color, BORDER_THICKNESS)


def draw_edge(canvas, source, target, color):
    adjacent = abs(source.x - target.x)
    opposite = abs(source.y - target.y)

    angle = 0 if adjacent == 0 else math.atan(opposite / adjacent)

    offset_x = int(math.cos(angle) * NODE_SIZE_RADIUS)
    offset_y = int(math.sin(angle) * NODE_SIZE_RADIUS)

    tail = [source.x, source.y]
    head = [target.x, target.y]

    if source.x > target.x:
        tail[0] -= offset_x
        head[0] += offset_x
    else:
        tail[0] += offset_x
        head[0] -= offset_x

    if source.y > target.y:
        tail[1] -= offset_y
        head[1] += offset_y
    else:
        tail[1] += offset_y
        head[1] -= offset_y

    cv2.line(canvas, tuple(tail), tuple(head), color, BORDER_THICKNESS, cv2.LINE_AA)


class OverlayDrawer():
    COLOR_CYCLE = 1

    def __init__(self, graph):
        self.graph = graph

    def calc_color(self):
        frequency = 0.3
        red   = math.sin(frequency * OverlayDrawer.COLOR_CYCLE + 0) * 127 + 128;
        green = math.sin(frequency * OverlayDrawer.COLOR_CYCLE + 2) * 127 + 128;
        blue  = math.sin(frequency * OverlayDrawer.COLOR_CYCLE + 4) * 127 + 128;

        OverlayDrawer.COLOR_CYCLE += 1
        return (blue, green, red)

    def draw(self, width, height):
        blank_image = np.zeros((width, height, 3), np.uint8)
        color = self.calc_color()

        for node in self.graph.nodes:
            draw_node(blank_image, node, color)

        for edge in self.graph.edges:
            if None in (edge.source, edge.target):
                continue

            source = self.graph.get_node_by_id(edge.source)
            target = self.graph.get_node_by_id(edge.target)
            draw_edge(blank_image, source, target, color)

        return blank_image
