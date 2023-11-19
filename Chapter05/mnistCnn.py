from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

from mnistData2 import MNIST


def build_model() -> Sequential:
    # Define the DNN
    model = Sequential()
    # Conv Block 1
    model.add(Conv2D(filters=32, kernel_size=(7, 7), input_shape=(28, 28, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Conv Block 2
    model.add(Conv2D(filters=64, kernel_size=(5, 5)))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output Layer
    model.add(Flatten())
    model.add(Dense(units=10))
    model.add(Activation("softmax"))
    return model


def main() -> None:
    mnist_data = MNIST()
    x_train, y_train = mnist_data.get_train_set()
    x_test, y_test = mnist_data.get_test_set()

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    model = build_model()
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    model.fit(x=x_train, y=y_train, epochs=30, batch_size=128)

    score = model.evaluate(x=x_test, y=y_test, batch_size=128)
    print(f"Test accuracy: {score[1]}")


if __name__ == "__main__":
    main()
