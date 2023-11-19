from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from mnistData1 import MNIST


def build_model() -> Sequential:
    model = Sequential()
    # (Input Layer and) Hidden Layer 1
    model.add(Dense(units=512, input_shape=(784,)))
    model.add(Activation("relu"))
    # Hidden Layer 2
    model.add(Dense(units=512))
    model.add(Activation("relu"))
    # Output Layer
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
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    model.fit(x=x_train, y=y_train, epochs=10, batch_size=128)

    score = model.evaluate(x=x_test, y=y_test, batch_size=128)
    print(f"Test accuracy: {score[0]}")


if __name__ == "__main__":
    main()
