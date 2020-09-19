from random import sample
import cv2
import numpy as np


class ImageRectifier():
    def __init__(self, img_path):
        self.n_lines = 64  # Number of lines from Hough transform
        self.min_scale = 0.10  # Only rectify if side of image is scaled by 1+-min_scale
        self.n_corner_samples = 256  # Number of line pairs for calculating new corners
        self.mind_coord_dist = 0.10  # Mind distance in fraction of img side for line pairs

        self.raw_img = img_path #cv2.imread(img_path)
        self.processed_img = self.preprocess()
        self.v_M = np.eye(3)
        self.h_M = np.eye(3)
        self.rectified_img = self.rectify_init()

    @property
    def img(self):
        return self.rectified_img

    def unrectify(self, img):
        unrectified_img = img.copy()
        unrectified_img = cv2.warpPerspective(
            unrectified_img, np.linalg.inv(self.v_M), (img.shape[1], img.shape[0]))
        unrectified_img = cv2.warpPerspective(
            unrectified_img, np.linalg.inv(self.h_M), (img.shape[1], img.shape[0]))
        return unrectified_img

    def rectify(self, img):
        rectified_img = img.copy()
        rectified_img = cv2.warpPerspective(
            rectified_img, self.h_M, (img.shape[1], img.shape[0]))
        rectified_img = cv2.warpPerspective(
            rectified_img, self.v_M, (img.shape[1], img.shape[0]))
        return rectified_img

    def rectify_init(self):
        self.n_lines = 64
        img = self.raw_img.copy()
        max_x = img.shape[1]
        max_y = img.shape[0]
        src = np.array(
            [(0, 0), (max_x, 0), (max_x, max_y), (0, max_y)], np.float32)

        h_lines = []
        h_lines_intersects = []
        lines = cv2.HoughLines(self.processed_img, 1, np.pi / 180.0, 10)
        for line in lines[:self.n_lines]:
            rho = line[0][0]
            theta = line[0][1]
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            if sin_t != 0:
                y_intersects = (rho / sin_t, (rho - cos_t * max_x) / sin_t)
                if 0 < y_intersects[0] < max_y and 0 < y_intersects[1] < max_y:
                    h_lines.append(line)
                    h_lines_intersects.append(y_intersects)
                    # self.draw_line(img, rho, theta, (0, 255, 0))
        right_corners_y = self.get_new_corners(max_y, h_lines_intersects)
        scale = max_y / (right_corners_y[1] - right_corners_y[0])
        if not 1.0 - self.min_scale < scale < 1.0 + self.min_scale:
            if scale > 1.0:
                h_dst = np.array([(0, 0),
                                  (max_x, 0),
                                  (max_x, right_corners_y[0]),
                                  (0, right_corners_y[1])],
                                 np.float32)
            else:
                left_corners_y = [-right_corners_y[0] *
                                  scale, right_corners_y[1] * scale]
                h_dst = np.array([(0, left_corners_y[0]),
                                  (max_x, 0),
                                  (max_x, max_y),
                                  (0, left_corners_y[1])],
                                 np.float32)
            self.h_M = cv2.getPerspectiveTransform(src, h_dst)
            img = cv2.warpPerspective(img, self.h_M, (max_x, max_y))
            self.processed_img = cv2.warpPerspective(
                self.processed_img, self.h_M, (max_x, max_y))

        v_lines = []
        v_lines_intersects = []
        lines = cv2.HoughLines(self.processed_img, 1, np.pi / 180.0, 10)
        for line in lines[:self.n_lines]:
            rho = line[0][0]
            theta = line[0][1]
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            if cos_t != 0:
                x_intersects = (rho / cos_t, (rho - sin_t * max_y) / cos_t)
                if 0 < x_intersects[0] < max_x and 0 < x_intersects[1] < max_x:
                    v_lines.append(line)
                    v_lines_intersects.append(x_intersects)
                    # self.draw_line(img, rho, theta, (0, 0, 255))
        bottom_corners_x = self.get_new_corners(max_x, v_lines_intersects)
        scale = max_x / (bottom_corners_x[1] - bottom_corners_x[0])
        self.v_M = np.eye(3)
        if not 1.0 - self.min_scale < scale < 1.0 + self.min_scale:
            if scale > 1.0:
                v_dst = np.array([(0, 0),
                                  (max_x, 0),
                                  (bottom_corners_x[1], max_y),
                                  (bottom_corners_x[0], max_y)],
                                 np.float32)
            else:
                top_corners_x = [-bottom_corners_x[0] *
                                 scale, bottom_corners_x[1] * scale]
                v_dst = np.array([(top_corners_x[0], 0),
                                  (top_corners_x[1], 0),
                                  (max_x, max_y),
                                  (0, max_y)],
                                 np.float32)
            self.v_M = cv2.getPerspectiveTransform(src, v_dst)
        img = cv2.warpPerspective(img, self.v_M, (max_x, max_y))
        return img

    def get_new_corners(self, max_coord, lines_intersects):
        corners_samples = []
        while len(corners_samples) < self.n_corner_samples:
            intersects = sample(lines_intersects, 2)
            d_top = abs(intersects[0][0] - intersects[1][0])
            d_bottom = abs(intersects[0][1] - intersects[1][1])
            if d_top < max_coord * self.mind_coord_dist or \
                    d_bottom < max_coord * self.mind_coord_dist:
                continue
            scale = d_bottom / d_top
            offset = intersects[0][0] - intersects[0][1] / scale
            corners_samples.append((offset, max_coord / scale + offset))
        return np.median(corners_samples, axis=0)

    def preprocess(self):
        img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (9, 9), 0)
        return cv2.Canny(blur, 50, 200)

    def draw_line(self, img, rho, theta, color):
        l = np.linalg.norm(np.array(img.shape))
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + l*(-b)), int(y0 + l*(a)))
        pt2 = (int(x0 - l*(-b)), int(y0 - l*(a)))
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    image_rectifier = ImageRectifier('data/examples_png/ex1_pic.png')
    cv2.imwrite('rectified.png', image_rectifier.img)
