import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DQN(tf.keras.Model):
    def __init__(
        self,
        state_shape: int,
        num_actions: int,
        learning_rate: float
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.internal_model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        input_state = Input(shape=self.state_shape)
        x = Dense(units=24)(input_state)
        x = Activation("relu")(x)
        x = Dense(units=24)(x)
        x = Activation("relu")(x)
        q_value_pred = Dense(self.num_actions)(x)
        model = Model(
            inputs=input_state,
            outputs=q_value_pred
        )
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def call(self, inputs: np.ndarray):
        return self.internal_model(inputs).numpy()

    def fit(self, states: np.ndarray, q_values: np.ndarray):
        self.internal_model.fit(
            x=states,
            y=q_values,
            verbose=0
        )

    def update_model(self, other_model: tf.keras.Model):
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path: str):
        self.internal_model.load_weights(path)

    def save_model(self, path: str):
        self.internal_model.save_weights(path)


if __name__ == "__main__":
    d = DQN(
        state_shape=4,
        num_actions=2,
        learning_rate=0.001
    )
    d.internal_model.summary()
