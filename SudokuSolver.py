import cv2
from math import sqrt
import numpy as np


class SudokuSolver:
    def __init__(self, numberRecognizer):
        self.numberRecognizer = numberRecognizer

    def imageToGrid(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255, 1, 1, 11, 2
        )

        vertices = self.getVertices(thres)

        if len(vertices) != 4:
            return image

        side = int(sqrt((vertices[0][0] - vertices[2][0])
                        ** 2 + (vertices[0][1] - vertices[2][1]) ** 2))

        new_vertices = [[0, 0], [side, 0], [0, side], [side, side]]

        result = self.perspectiveSquareTransform(
            image, vertices,
            new_vertices,
            side, side
        )

        return self.getGridNumbers(result, side)

    def perspectiveSquareTransform(self, image, old_vertices, new_vertices, width, height):
        matrix = cv2.getPerspectiveTransform(
            np.float32(old_vertices),
            np.float32(new_vertices)
        )

        return cv2.warpPerspective(image, matrix, (width, height))

    def getVertices(self, image):
        vertices_ = self.getBestContour(image)
        vertices = []
        for v in vertices_:
            vertices.append([v[0][0], v[0][1]])

        vertices = sorted(vertices, key=lambda a: a[0]+a[1])

        if len(vertices) == 4 and (vertices[2][0] > vertices[1][0]):
            vertices[1], vertices[2] = vertices[2], vertices[1]

        return vertices

    def getBestContour(self, image):
        contours, _ = cv2.findContours(
            image, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000 and area > max_area:
                max_area = area
                best_cnt = contour

        epsilon = 0.06 * cv2.arcLength(best_cnt, True)

        return cv2.approxPolyDP(best_cnt, epsilon, True)

    def getGridNumbers(self, image, side):
        grid_size = 9
        step = side / grid_size

        for x in range(grid_size):
            for y in range(grid_size):
                x_step = int(x * step)
                y_step = int(y * step)
                crop_step = int(step - 5)

                croped_img = cv2.resize(
                    image[x_step + 5:x_step+crop_step,
                          y_step + 5:y_step+crop_step],
                    (50, 50))

                # cv2.imwrite('./output/'+str(self.c)+str(x)+'_'+str(y)+'.png', croped_img)

        return image
