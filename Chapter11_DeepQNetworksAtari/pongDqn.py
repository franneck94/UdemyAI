import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import RMSprop


class DQN(tf.keras.Model):
    def __init__(self, img_shape: tuple, num_actions: int, learning_rate: float) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.loss = Huber()
        self.optimizer = RMSprop(
            lr=0.00025,
            rho=0.95,
            epsilon=0.01
        )
        self.internal_model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        img = Input(shape=self.img_shape)
        x = Conv2D(filters=32, kernel_size=8, strides=4, padding="same")(img)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        out = Dense(self.num_actions)(x)
        model = Model(inputs=img, outputs=out)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def call(self, inputs: np.ndarray) -> np.ndarray:
        return self.internal_model(inputs).numpy()

    def fit(self, states: np.ndarray, q_values: np.ndarray) -> None:
        self.internal_model.fit(x=states, y=q_values, verbose=0)

    def update_model(self, other_model: tf.keras.Model) -> None:
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path: str) -> None:
        self.internal_model.load_weights(path)

    def save_model(self, path: str) -> None:
        self.internal_model.save_weights(path)


if __name__ == "__main__":
    dqn = DQN(
        img_shape=(64, 64, 4),
        num_actions=2,
        learning_rate=0.001
    )
    dqn.internal_model.summary()
