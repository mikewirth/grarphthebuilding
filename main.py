import cv2
import numpy as np

from random import sample
# from sklearn.cluster import MeanShift


class AwesomeClass():
    def __init__(self):
        self.orig_img = self.load('data/examples_png/ex1_pic.png')
        self.raw_img = self.load('data/examples_png/ex1_pic.png')
        self.bw_img = self.preprocess()
        # self.contours = self.get_contours()
        self.rectify()

        cv2.imwrite('image.png', np.concatenate(
            (self.orig_img, self.raw_img), axis=0))

    def rectify(self):
        n_lines = 64
        img = self.raw_img
        max_x = img.shape[1]
        max_y = img.shape[0]
        src = np.array(
            [(0, 0), (max_x, 0), (max_x, max_y), (0, max_y)], np.float32)

        h_lines = []
        h_lines_intersects = []
        lines = cv2.HoughLines(self.bw_img, 1, np.pi / 180.0, 100)
        for line in lines[:n_lines]:
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
        if 0.90 < scale < 1.10:
            return
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
        h_M = cv2.getPerspectiveTransform(src, h_dst)
        self.raw_img = cv2.warpPerspective(self.raw_img, h_M, (max_x, max_y))

        v_lines = []
        v_lines_intersects = []
        lines = cv2.HoughLines(self.bw_img, 1, np.pi / 180.0, 100)
        for line in lines[:n_lines]:
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
        if 0.90 < scale < 1.10:
            return
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
        v_M = cv2.getPerspectiveTransform(src, v_dst)
        self.raw_img = cv2.warpPerspective(self.raw_img, v_M, (max_x, max_y))

    def get_new_corners(self, max_coord, lines_intersects):
        n_samples = 256
        corners_samples = []
        while len(corners_samples) < n_samples:
            intersects = sample(lines_intersects, 2)
            d_top = abs(intersects[0][0] - intersects[1][0])
            d_bottom = abs(intersects[0][1] - intersects[1][1])
            if d_top < max_coord / 10.0 or d_bottom < max_coord / 10.0:
                continue
            scale = d_bottom / d_top
            offset = intersects[0][0] - intersects[0][1] / scale
            corners_samples.append((offset, max_coord / scale + offset))
        return np.median(corners_samples, axis=0)

    # def get_contours(self):
    #     contours, _ = cv2.findContours(
    #         self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = [contour for contour in contours if cv2.arcLength(
    #         contour, False) > 4]
    #     contours.sort(key=lambda x: cv2.arcLength(x, True), reverse=True)
    #     self.img = np.zeros(self.img.shape, dtype=np.uint8)
        # cv2.drawContours(self.img, contours, 0, (255, 255, 255), 2)
        return contours

    def preprocess(self):
        img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        # Blur
        blur = cv2.GaussianBlur(img, (9, 9), 0)
        # Laplace
        dst = cv2.Laplacian(blur, cv2.CV_8UC1, ksize=5)
        # Otsu thresholding
        _, thresh = cv2.threshold(
            dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def load(self, img_path):
        return cv2.imread(img_path)
        # return cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    def draw_line(self, img, rho, theta, color):
        l = np.linalg.norm(np.array(img.shape))
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + l*(-b)), int(y0 + l*(a)))
        pt2 = (int(x0 - l*(-b)), int(y0 - l*(a)))
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)


def main():
    AwesomeClass()


if __name__ == "__main__":
    main()
