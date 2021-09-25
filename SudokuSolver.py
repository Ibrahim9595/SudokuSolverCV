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

        vertices = self._getVertices(thres)

        if len(vertices) != 4:
            return image, vertices

        side = int(sqrt((vertices[0][0] - vertices[2][0])
                        ** 2 + (vertices[0][1] - vertices[2][1]) ** 2))

        new_vertices = [[0, 0], [side, 0], [0, side], [side, side]]

        result = self.__perspectiveSquareTransform(
            image, vertices,
            new_vertices,
            side, side
        )

        return (result, self.__getGridNumbers(result, side))

    def __perspectiveSquareTransform(self, image, old_vertices, new_vertices, width, height):
        matrix = cv2.getPerspectiveTransform(
            np.float32(old_vertices),
            np.float32(new_vertices)
        )

        return cv2.warpPerspective(image, matrix, (width, height))

    def _getVertices(self, image):
        vertices_ = self.__getBestContour(image)
        vertices = []
        for v in vertices_:
            vertices.append([v[0][0], v[0][1]])

        vertices = sorted(vertices, key=lambda a: a[0]+a[1])

        if len(vertices) == 4 and (vertices[2][0] > vertices[1][0]):
            vertices[1], vertices[2] = vertices[2], vertices[1]

        return vertices

    def __getBestContour(self, image):
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

    def __getGridNumbers(self, image, side):
        grid_size = 9
        step = side / grid_size
        croped_imgs = []
        labels = []

        for x in range(grid_size):
            for y in range(grid_size):
                x_step = int(x * step)
                y_step = int(y * step)
                crop_step = int(step - 7)

                croped_img = cv2.resize(
                    image[x_step + 7:x_step+crop_step,
                          y_step + 7:y_step+crop_step],
                    (50, 50))

                croped_imgs.append(croped_img)
                labels.append((x, y))

        predictions = self.numberRecognizer.recognize_numbers(
            croped_imgs, labels
        )


        return self.__createGrid(predictions)

    def __createGrid(self, predictions):
        grid = [[0 for i in range(9)] for j in range(9)]

        for prediction in predictions:
            (x, y) = prediction[1]
            grid[x][y] = prediction[0]

        return grid

    def showNumbersOnImage(self, image, grid, step):
        for x_ in range(len(grid)):
            for y_ in range(len(grid[x_])):
                x = (int(x_ * step) + int(x_ * step + step)) // 2
                y = (int(y_ * step) + int(y_ * step + step)) // 2

                image = cv2.putText(
                    image, str(grid[x_][y_]),
                    (y, x), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3, cv2.LINE_AA
                )

        cv2.imshow('Test', image)
