import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Actor(Model):
    def __init__(self, num_observations, num_actions, num_values, learning_rate_actor):
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_values = num_values
        self.learning_rate_actor = learning_rate_actor

        state = Input(shape=(num_observations,))
        x = Dense(24)(state)
        x = Activation("relu")(x)
        x = Dense(self.num_actions)(x)
        actor_out = Activation("softmax")(x)
        self.internal_model = Model(
            inputs=state,
            outputs=actor_out
        )
        self.internal_model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate_actor)
        )

    def call(self, states: np.ndarray) -> np.ndarray:
        return self.internal_model(states).numpy()

    def fit(self, states: np.ndarray, actions: np.ndarray):
        self.internal_model.fit(x=states, y=actions, verbose=0)

    def update_model(self, other_model: Model):
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path: str):
        self.internal_model.load_weights(path)

    def save_model(self, path: str):
        self.internal_model.save_weights(path)


class Critic(Model):
    def __init__(self, num_observations, num_actions, num_values, learning_rate_critic):
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_values = num_values
        self.learning_rate_critic = learning_rate_critic

        state = Input(shape=(num_observations,))
        x = Dense(24)(state)
        x = Activation("relu")(x)
        x = Dense(self.num_values)(x)
        critic_out = Activation("linear")(x)
        self.internal_model = Model(
            inputs=state,
            outputs=critic_out
        )
        self.internal_model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate_critic)
        )

    def call(self, states: np.ndarray) -> np.ndarray:
        return self.internal_model(states).numpy()

    def fit(self, states: np.ndarray, values: np.ndarray):
        self.internal_model.fit(x=states, y=values, verbose=0)

    def update_model(self, other_model: Model):
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path: str):
        self.internal_model.load_weights(path)

    def save_model(self, path: str):
        self.internal_model.save_weights(path)
