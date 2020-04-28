# from SudokuSolver import SudokuSolver

# solver = SudokuSolver(None)


# paths = [
#     "./inputs/a.jpg",
#     "./inputs/b.jpg",
#     "./inputs/c.jpg",
#     "./inputs/d.jpg",
#     "./inputs/e.jpg",
#     "./inputs/f.jpg",
#     "./inputs/g.jpg",
#     "./inputs/h.jpg",
#     "./inputs/i.jpeg",
#     "./inputs/j.jpg",
#     "./inputs/k.jpg",
#     "./inputs/l.jpeg",
#     "./inputs/m.jpg",
#     "./inputs/n.png"
# ]

# for path in paths:
#     img = solver.imageToGrid(cv2.imread(path))

#     cv2.imshow('test', cv2.resize(img, (300, 300)))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

from NumberRecognizer import NumberRecognizer
import cv2
from os import listdir

root = './output/'
paths = listdir(root)
images = []
labels = []
img_dict = {}

for path in paths:
    images.append(cv2.imread(root + path))
    label = tuple(path.split('/')[-1].split('.')[0].split('_'))
    labels.append(label)
    img_dict[label] = images[-1]


recognizer = NumberRecognizer('./deeplearning/models/cnn/best_model')

results = recognizer.recognize_numbers(images, labels)

for r in results:
    print(r[0])
    cv2.imshow('test', img_dict[r[1]])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
