from joblib import load
from skimage.feature import hog
import cv2

class KNNRecognizer:
    def __init__(self, model_path):
        self.model = load(model_path)

    def recognize_numbers(self, images, labels):
        features = []
        ret = []
        for image in images:
            features.append(hog(cv2.resize(image, (50, 50)), orientations=8,
                                pixels_per_cell=(4, 4), cells_per_block=(7, 7)))
        
        predictions = self.model.predict(features)

        for i in range(len(predictions)):
            val = predictions[i]
            ret.append((val, labels[i]))

        return ret
