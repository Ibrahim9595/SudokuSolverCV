from Recognizers.CNNRecognizer import CNNRecognizer
# from Recognizers.KNNRecognizer import KNNRecognizer


class NumberRecognizer:
    def __init__(self, model_path, type='CNN'):
        if type == 'CNN':
            self.model = CNNRecognizer(model_path)
        # else:
            # self.model = KNNRecognizer(model_path)

    def recognize_numbers(self, images, labels):
        return self.model.recognize_numbers(images, labels)
