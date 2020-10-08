import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


class MNIST:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # rescale values
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    def data_augmentation(self, augment_size=5000):
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False,
            data_format="channels_last",
            zca_whitening=True
        )
        # fit data for zca whitening
        image_generator.fit(self.x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[randidx].copy()
        y_augmented = self.y_train[randidx].copy()
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,
            shuffle=False
        ).next()[0]
        # append augmented data to trainset
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

    def get_train_set(self):
        return (self.x_train, self.y_train)

    def get_test_set(self):
        return (self.x_test, self.y_test)
