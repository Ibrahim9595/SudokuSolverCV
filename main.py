from SudokuSolver import SudokuSolver
from NumberRecognizer import NumberRecognizer
from SudokuAlgorithm import SudokuAlgorithm
import cv2

recognizer = NumberRecognizer('./deeplearning/models/cnn/best_model')
solver = SudokuSolver(recognizer)


paths = [
    # "./inputs/c.jpg",
    # "./inputs/d.jpg",
    # "./inputs/e.jpg",
    # "./inputs/g.jpg",
    # "./inputs/n.png",
    # "./inputs/o.png",
    # "./inputs/p.png",
    # "./inputs/q.png",
    # "./inputs/r.png",
    # "./inputs/s.png",
    # "./inputs/u.png",
    # "./inputs/v.png",
    # "./inputs/x.png",
    "./inputs/fail/y.jpeg",
]

for path in paths:
    try:
        img, grid = solver.imageToGrid(cv2.imread(path))
        img = cv2.resize(img, (600, 600))
        h, w, ch = img.shape

        if grid != None:
            algorithm = SudokuAlgorithm(grid)
            solution = algorithm.solve()
            if solution != None:
                solver.showNumbersOnImage(img, solution, h // 9)

    except:
        print('error in ' + path)
        solver.showNumbersOnImage(img, grid, h // 9)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
