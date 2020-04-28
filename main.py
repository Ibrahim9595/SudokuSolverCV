from SudokuSolver import SudokuSolver
from NumberRecognizer import NumberRecognizer
import cv2

recognizer = NumberRecognizer('./deeplearning/models/cnn/best_model')
solver = SudokuSolver(recognizer)


paths = [
    # "./inputs/a.jpg",
    # "./inputs/b.jpg",
    # "./inputs/c.jpg",
    # "./inputs/d.jpg",
    # "./inputs/e.jpg",
    # "./inputs/f.jpg",
    # "./inputs/g.jpg",
    # "./inputs/h.jpg",
    # "./inputs/i.jpeg",
    # "./inputs/j.jpg",
    # "./inputs/k.jpg",
    # "./inputs/l.jpeg",
    # "./inputs/m.jpg",
    # "./inputs/n.png",
    "./inputs/o.png"
]

for path in paths:
    img = solver.imageToGrid(cv2.imread(path))

    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
