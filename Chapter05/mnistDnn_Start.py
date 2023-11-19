from mnistData1 import MNIST


def main() -> None:
    mnist_data = MNIST()
    x_train, y_train = mnist_data.get_train_set()
    x_test, y_test = mnist_data.get_test_set()

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")


if __name__ == "__main__":
    main()
