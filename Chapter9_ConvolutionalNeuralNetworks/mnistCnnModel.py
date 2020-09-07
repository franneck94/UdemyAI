import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *

from mnistData import *


mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()

# Define the CNN
model = Sequential()
# Conv Block 1
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1))) # 28x28x32
model.add(Conv2D(32, (3, 3))) # 28x28x32
model.add(MaxPooling2D(pool_size=(2, 2))) # 14x14x32
model.add(Activation("relu")) # 14x14x32
# Dense Block
model.add(Flatten()) # 6272
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

# Print the CNN layers
model.summary()

# Train the CNN
lr = 0.0001
optimizer = Adam(lr=lr)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
model.fit(
    x_train,
    y_train,
    verbose=1,
    batch_size=128,
    epochs=1,
    validation_data=(x_test, y_test))

# Test the CNN
score = model.evaluate(
    x_test,
    y_test,
    verbose=0)
print("Test accuracy: ", score[1])
