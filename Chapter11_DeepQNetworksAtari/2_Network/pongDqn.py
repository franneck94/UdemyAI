from keras.models import *
from keras.layers import *
from keras.optimizers import *

import tensorflow as tf


class DQN(Model):
    def __init__(self, img_shape, num_actions, lr):
        super(DQN, self).__init__()
        self.img_shape = img_shape
        self.num_actions = num_actions
        self.lr = lr
        self.model = self.build_model()
        self.loss = tf.losses.Huber()
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.lr))

    def build_model(self):
        img = Input(shape=self.img_shape)
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding="same")(img)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        x = Dense(64)(x)
        x = Activation("relu")(x)
        out = Dense(self.num_actions)(x)
        model = Model(inputs=img, outputs=out)
        return model

    def train(self, states, q_values):
        self.model.fit(states, q_values, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def update_model(self, other_model):
        self.model.set_weights(other_model.get_weights())

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)


if __name__ == "__main__":
    d = DQN(img_shape=(84, 84, 4), num_actions=2, lr=0.001)
    d.model.summary()
