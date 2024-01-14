from mnistData1 import MNIST


def main() -> None:
    mnist_data = MNIST()
    x_train, y_train = mnist_data.get_train_set()  # noqa: F841
    x_test, y_test = mnist_data.get_test_set()  # noqa: F841

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")


if __name__ == "__main__":
    main()
