from numpy import argmax
from tensorflow.keras import models
from tensorflow import (
    image as tf_image,
    float32 as tf_float32,
    convert_to_tensor
)


class NumberRecognizer:
    def __init__(self, model_path):
        self.model = models.load_model(model_path)
        _, self.image_width, self.image_height, _ = self.model.layers[0].input_shape

    def decode_img(self, img):
        # convert the compressed string to a 3D gray_scale_img
        img = tf_image.rgb_to_grayscale(img)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf_image.convert_image_dtype(img, tf_float32)
        # resize the image to the desired size.
        return tf_image.resize(img, [self.image_width, self.image_height])

    def recognize_numbers(self, images, labels):
        p_images = []

        for image in images:
            p_images.append(self.decode_img(image))

        images = convert_to_tensor(p_images, dtype=tf_float32)

        predictions = self.model.predict(images)

        ret = []

        for i in range(len(predictions)):
            val = argmax(predictions[i])
            if val != 0:
                ret.append((val, labels[i]))

        return ret
