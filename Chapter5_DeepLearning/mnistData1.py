from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class MNIST:
    def __init__(self):
        (self.x_train, self.y_train), (
            self.x_test,
            self.y_test,
        ) = mnist.load_data()
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 784)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 784)
        # convert from int to float
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")
        # rescale values
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    def get_train_set(self):
        return (self.x_train, self.y_train)

    def get_test_set(self):
        return (self.x_test, self.y_test)
