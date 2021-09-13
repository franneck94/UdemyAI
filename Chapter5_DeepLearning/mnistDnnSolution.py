from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from mnistData1 import MNIST


mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# Define the DNN
model = Sequential()
# Hidden Layer 1
model.add(Dense(units=512, input_shape=(784,)))
model.add(Activation("relu"))
# Hidden Layer 2
model.add(Dense(units=256))
model.add(Activation("relu"))
# Hidden Layer 3
model.add(Dense(units=128))
model.add(Activation("relu"))
# Output Layer
model.add(Dense(units=10))
model.add(Activation("softmax"))

# Print the DNN layers
model.summary()

# Compile the DNN
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"],
)

# Train the DNN
model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=128,
    epochs=30,
    validation_data=(x_test, y_test),
)

# Test the DNN
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {score}")
