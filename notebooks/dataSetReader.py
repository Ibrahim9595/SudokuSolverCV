import tensorflow as tf
import numpy as np
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSetReader:
    def __init__(self, data_dir='./dataset/train', batch_size=32, img_height=50, img_width=50):
        self.BATCH_SIZE = batch_size
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width

        data_dir = pathlib.Path(data_dir)
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
        image_count = len(list(data_dir.glob('*/*')))

        self.STEPS_PER_EPOCH = np.ceil(image_count / self.BATCH_SIZE)
        self.CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

        self.labeled_ds = list_ds.map(
            self.process_path, num_parallel_calls=AUTOTUNE)

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        return int(parts[-2])

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=3)
        # convert the compressed string to a 3D gray_scale_img
        img = tf.image.rgb_to_grayscale(img)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self, cache=True, shuffle_buffer_size=1000):
        ds = self.labeled_ds.batch(self.BATCH_SIZE)

        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def show_batch(self, image_batch, label_batch):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 16))

        for n in range(32):
            ax = plt.subplot(4, 8, n+1)
            plt.imshow(image_batch[n], cmap=plt.cm.gray)
            plt.title(str(label_batch[n]))
            plt.axis('off')

        plt.show()
