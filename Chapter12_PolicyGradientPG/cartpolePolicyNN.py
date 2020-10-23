import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class NN(Model):
    def __init__(self, state_shape, num_actions, learning_rate):
        super(NN, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        state = Input(shape=(state_shape,))
        x = Dense(24)(state)
        x = Activation("relu")(x)
        x = Dense(self.num_actions)(x)
        out = Activation("softmax")(x)
        self.internal_model = Model(
            inputs=state,
            outputs=out
        )
        self.internal_model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate)
        )

    def call(self, inputs: np.ndarray) -> np.ndarray:
        return self.internal_model(inputs).numpy()

    def fit(self, states: np.ndarray, q_values: np.ndarray):
        self.internal_model.fit(x=states, y=q_values, verbose=0)

    def update_model(self, other_model: Model):
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path: str):
        self.internal_model.load_weights(path)

    def save_model(self, path: str):
        self.internal_model.save_weights(path)
